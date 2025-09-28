import argparse, os, random, json
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torch.amp import autocast, GradScaler  # <-- AMP (new API)
from torch.optim.swa_utils import AveragedModel, update_bn  # (only if you try SWA later; safe to import)
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, classification_report
import csv

import gc, torch

# optional: timm for NASNet / Xception
try:
    import timm
    HAVE_TIMM = True
except Exception:
    HAVE_TIMM = False

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    return [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]

# ----------------------------
# (3) Animal cropper using Faster R-CNN (COCO 'cow' class=20)
# ----------------------------
def crop_animals(src, dst, score_thr=0.7, pad=0.06, batch_det=4):
    """
    Batched Faster R-CNN inference on GPU (if available) with AMP.
    Falls back to square center crop when no suitable box is found.
    """
    device = get_device()
    weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    print("Cropping on device:", device)

    # Accept several large-animal classes (helps when buffalo/cow get mislabeled)
    cats = weights.meta.get("categories", [])
    name_to_id = {n:i for i,n in enumerate(cats)}
    wanted = {n for n in ["cow","horse","sheep","zebra","giraffe","elephant"]}
    WANTED = {name_to_id[n] for n in wanted if n in name_to_id}

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for cls_dir in [d for d in src.iterdir() if d.is_dir()]:
        out_dir = dst / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        files = list_images(cls_dir)
        print(f"{cls_dir.name}: {len(files)} imgs")

        # simple batching
        for i in range(0, len(files), batch_det):
            batch_paths = files[i:i+batch_det]
            ims = []
            pil_ims = []
            for p in batch_paths:
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    continue
                pil_ims.append((p, im))
                ims.append(TF.to_tensor(im))
            if not ims:
                continue

            # inference (AMP via torch.amp.autocast)
            with torch.inference_mode():
                with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
                    outputs = model([t.to(device) for t in ims])

            # postprocess each image in the batch
            for (p, im), out in zip(pil_ims, outputs):
                w, h = im.size
                boxes = out["boxes"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()

                best = None; best_area = 0
                for b, lab, sc in zip(boxes, labels, scores):
                    if lab in WANTED and sc >= score_thr:
                        x1,y1,x2,y2 = b
                        area = max(0,(x2-x1))*max(0,(y2-y1))
                        if area > best_area:
                            best_area = area; best = (x1,y1,x2,y2)

                if best is None:
                    # center square fallback
                    short = min(w,h); left=(w-short)//2; top=(h-short)//2
                    crop = im.crop((left, top, left+short, top+short))
                else:
                    x1,y1,x2,y2 = best
                    dx = (x2-x1)*pad; dy = (y2-y1)*pad
                    x1 = max(0, int(x1-dx)); y1 = max(0, int(y1-dy))
                    x2 = min(w, int(x2+dx)); y2 = min(h, int(y2+dy))
                    crop = im.crop((x1,y1,x2,y2))

                crop.save(out_dir / Path(p).name)

# ----------------------------
# Datasets & transforms
# ----------------------------
def make_transforms(img_size, train=True, strong=True):
    if train:
        aug = [
            # Keep most of the animal in-frame; avoid extreme crops/aspect warps
            transforms.Resize(int(img_size * 1.05)),
            transforms.RandomResizedCrop(img_size, scale=(0.80, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(p=0.5),

            # Small, realistic geometry tweaks (rotation+slight translate/scale/shear)
            transforms.RandomAffine(
                degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=2
            ),

            # Softer color jitter (keeps coat texture/tones usable)
            transforms.ColorJitter(0.10, 0.10, 0.10, 0.02),
        ]

        # Optional tiny robustness—kept mild
        if strong:
            aug += [
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.10),
            ]

        aug += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

        # Place RandomErasing last and tone it down; it can hurt fine-grained cues
        aug += [
            transforms.RandomErasing(
                p=0.10 if strong else 0.05,
                scale=(0.02, 0.06),
                ratio=(0.3, 3.3),
                value="random",
            )
        ]
        return transforms.Compose(aug)

    else:
        # Val/test: a touch more resize before center crop for stability
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.10)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

def make_loaders(
    root,
    img_size,
    batch,
    val_split=0.15,
    num_workers=2,
    balanced=True,
    # NEW: focus options (names are case/space tolerant, e.g. "red sindhi" or red_sindhi)
    focus_classes=None,
    focus_factor=2.0,
    # NEW: dataloader QoL
    prefetch_factor=4,
    persistent=True,
):
    """
    Build train/val loaders with optional class rebalancing and extra up-weighting
    for specific 'focus' classes by name.

    Example:
      train_loader, val_loader, classes = make_loaders(
          "data/crops/combined_train_d1_d3", 320, 32,
          num_workers=12, balanced=True,
          focus_classes=["red sindhi","krishna valley","kenkatha","nimari","ongole","alambadi"],
          focus_factor=2.0
      )
    """
    # ------------- scan labels once
    base = datasets.ImageFolder(root, transform=transforms.ToTensor())
    classes = base.classes                # e.g. ["Alambadi", "Bargur", ...]
    y = [s[1] for s in base.samples]      # numeric labels

    # map normalized class name -> class index
    import re
    def _norm(s: str) -> str:
        s = s.strip().lower().replace("_", " ").replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s
    name_to_idx = {_norm(c): i for i, c in enumerate(classes)}

    # ------------- split (stratified)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))

    # ------------- real datasets with proper transforms
    t_train = make_transforms(img_size, train=True)
    t_val   = make_transforms(img_size, train=False)
    train_ds = torch.utils.data.Subset(datasets.ImageFolder(root, transform=t_train), train_idx)
    val_ds   = torch.utils.data.Subset(datasets.ImageFolder(root, transform=t_val),   val_idx)

    # ------------- sampler (class balance + optional focus boost)
    if balanced:
        # inverse-frequency base weights
        train_labels = [y[i] for i in train_idx]
        from collections import Counter
        counts = Counter(train_labels)                     # per-class counts in train split
        class_w = {c: 1.0 / max(1, counts[c]) for c in counts}

        weights = [class_w[lab] for lab in train_labels]  # per-sample weights

        # optional: up-weight focus classes by name
        if focus_classes:
            # build set of label indices to boost
            focus_set = set()
            for name in focus_classes:
                key = _norm(name)
                if key in name_to_idx:
                    focus_set.add(name_to_idx[key])
            if focus_set:
                for j, lab in enumerate(train_labels):
                    if lab in focus_set:
                        weights[j] *= float(focus_factor)

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # ------------- loaders (fast defaults)
    pin = torch.cuda.is_available()
    persistent_workers = (num_workers > 0 and persistent)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch//2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )

    return train_loader, val_loader, classes

# ----------------------------
# (1) Transfer models
# ----------------------------

def build_model(backbone, num_classes):
    # torchvision backbones you already had
    if backbone == "effb0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)

    elif backbone == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)

    # --- NEW: torchvision DenseNet & MobileNet
    elif backbone == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, num_classes)

    elif backbone == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, num_classes)

    # --- If you’ve already added these earlier, keep them:
    elif backbone == "effv2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)

    elif backbone == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        in_f = m.head.in_features
        m.head = nn.Linear(in_f, num_classes)

    elif backbone == "vit_b16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f, num_classes)

    # --- NEW: NASNet / Xception via timm
    elif backbone in {"nasnetalarge", "nasnetamobile", "xception"}:
        assert HAVE_TIMM, \
            f"Backbone '{backbone}' needs timm. Install: pip install timm"
        m = timm.create_model(backbone, pretrained=True, num_classes=num_classes)

    else:
        raise ValueError("backbone must be one of: "
                         "effb0, convnext_tiny, effv2_s, swin_t, vit_b16, "
                         "densenet121, mobilenet_v3_large, nasnetalarge, nasnetamobile, xception")
    return m

# ----------------------------
# (6) MixUp / CutMix (simple)
# ----------------------------
def rand_beta(alpha=0.4):
    return np.random.beta(alpha, alpha) if alpha > 0 else 1.0

def apply_mixup_cutmix(x, y, p=0.5, alpha=0.4):
    if random.random() > p:
        return x, y, None  # no mix
    lam = rand_beta(alpha)
    perm = torch.randperm(x.size(0))
    x2, y2 = x[perm], y[perm]
    if random.random() < 0.5:
        # MixUp
        x_mixed = lam * x + (1-lam) * x2
        return x_mixed, (y, y2), lam
    else:
        # CutMix (rectangle)
        B, C, H, W = x.size()
        cx, cy = np.random.randint(W), np.random.randint(H)
        w = int(W * np.sqrt(1-lam)); h = int(H * np.sqrt(1-lam))
        x1, y1 = np.clip(cx - w//2, 0, W), np.clip(cy - h//2, 0, H)
        x2_, y2_ = np.clip(cx + w//2, 0, W), np.clip(cy + h//2, 0, H)
        x_mixed = x.clone()
        x_mixed[:, :, y1:y2_, x1:x2_] = x2[:, :, y1:y2_, x1:x2_]
        lam_adj = 1 - ((x2_-x1)*(y2_-y1)/(W*H))
        return x_mixed, (y, y2), lam_adj

def loss_with_mix(crit, logits, target):
    # target is either y or (y1, y2)
    if isinstance(target, tuple):
        y1, y2 = target[0], target[1]
        lam = target[2] if len(target) > 2 else 0.5
        return lam*crit(logits, y1) + (1-lam)*crit(logits, y2)
    return crit(logits, target)

# ----------------------------
# Train / Eval helpers
# ----------------------------
def accuracy_top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, device, crit, tta=False, amp_enabled=False, amp_dtype=torch.float16):
    model.eval()
    tot_loss=tot_acc=n=0
    for xb, yb in tqdm(loader, desc="val", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            if tta:
                # simple TTA: normal + hflip
                logits1 = model(xb)
                xb_flip = torch.flip(xb, dims=[3])
                logits2 = model(xb_flip)
                logits = (logits1 + logits2)/2
            else:
                logits = model(xb)
            loss = crit(logits, yb)
        bs = yb.size(0)
        tot_loss += loss.item()*bs
        tot_acc  += accuracy_top1(logits, yb)*bs
        n += bs
    return tot_loss/n, tot_acc/n

def train_one_stage(model, loaders, device, epochs=5, lr=3e-4, weight_decay=1e-4,
                    label_smoothing=0.1, mix_p=0.5, mix_alpha=0.4, ckpt_path=None,
                    amp_enabled=False, amp_dtype=torch.float16, use_scaler=False,
                    opt_name="adamw", scheduler_kind="plateau"):
    train_loader, val_loader = loaders

    # Loss (tunable smoothing)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_kind == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        step_on_val = False
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, verbose=True)
        step_on_val = True

    # AMP scaler (only needed for fp16)
    scaler = GradScaler(enabled=use_scaler)

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = tot_acc = n = 0
        for xb, yb in tqdm(train_loader, desc=f"train (ep {ep})", leave=False):
            xb, yb = xb.to(device), yb.to(device)

            # MixUp / CutMix (mild by default)
            xb2, target, lam = apply_mixup_cutmix(xb, yb, p=mix_p, alpha=mix_alpha)
            if lam is not None:  # pack lambda for the loss
                target = (target[0], target[1], lam)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(xb2)
                loss = loss_with_mix(crit, logits, target)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs = yb.size(0)
            tot_loss += loss.item() * bs
            # accuracy vs. original labels (approx when mixed)
            tot_acc  += (logits.detach().argmax(1) == yb).float().sum().item()
            n += bs

        tr_loss, tr_acc = tot_loss / n, tot_acc / n

        try:
            del xb, yb, xb2, logits, loss  # they still exist outside the loop scope
        except NameError:
            pass
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # Validation
        va_loss, va_acc = evaluate(
            model, val_loader, device, crit, tta=False,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype
        )

        # Scheduler step
        if step_on_val:
            sched.step(va_acc)     # ReduceLROnPlateau
        else:
            sched.step()           # CosineAnnealing

        # Logging
        cur_lr = opt.param_groups[0]["lr"]
        print(f"  lr {cur_lr:.2e} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        # Save best checkpoint by val accuracy
        if ckpt_path and va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"  ✔ saved best: {ckpt_path}")

    return best_acc

@torch.no_grad()
def cmd_confmat(args):
    """
    Build a confusion matrix for a dataset using one or more checkpoints (ensemble).
    Saves CSV (and optional PNG heatmap), and prints the 10 weakest classes.
    """
    device = get_device()
    print("device:", device)

    # --- training class list from first ckpt (defines global order)
    meta_path0 = Path(args.ckpt[0]).with_name("meta.json")
    assert meta_path0.exists(), f"meta.json not found next to {args.ckpt[0]}"
    meta0 = json.load(open(meta_path0))
    train_classes = meta0["classes"]
    num_train = len(train_classes)
    train_class_to_idx = {c:i for i,c in enumerate(train_classes)}

    # --- eval dataset (must be subset of training classes)
    tfm = make_transforms(args.size, train=False)
    ds = datasets.ImageFolder(args.data, transform=tfm)
    eval_classes = ds.classes

    # ensure eval classes ⊆ train classes (and build mapping in same order as ImageFolder)
    missing = [c for c in eval_classes if c not in train_class_to_idx]
    if missing:
        raise ValueError(f"These eval classes are not in training classes: {missing}\n"
                         f"Make sure names match or drop those folders before running.")

    subset_to_train = torch.tensor([train_class_to_idx[c] for c in eval_classes], device=device)  # (C_eval,)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    # --- load ensemble
    models_list = []
    for ck in args.ckpt:
        meta_path = Path(ck).with_name("meta.json")
        backbone = "effb0"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            backbone = meta.get("backbone", backbone)
            if meta.get("classes") and meta["classes"] != train_classes:
                print(f"Warning: class list in {ck} differs; using the first checkpoint's ordering.")
        m = build_model(backbone, num_classes=num_train).to(device)
        state = torch.load(ck, map_location=device)
        m.load_state_dict(state["model"])
        m.eval()
        models_list.append(m)
        print(f"Loaded {backbone} from {ck}")

    # --- forward pass (AMP + TTA), restrict logits to eval classes
    amp_enabled = (device.type == "cuda")
    y_true, y_pred = [], []
    for xb, yb in tqdm(loader, desc="confmat eval"):
        xb, yb = xb.to(device), yb.to(device)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits_agg = 0
            for m in models_list:
                logits = m(xb)
                if args.tta:
                    logits = (logits + m(torch.flip(xb, dims=[3]))) / 2
                logits_agg = logits_agg + logits
            logits_agg = logits_agg / len(models_list)
            # keep only eval classes (columns reordered to eval order)
            logits_eval = logits_agg.index_select(dim=1, index=subset_to_train)
            preds = logits_eval.argmax(1)

        y_true.extend(yb.tolist())   # indices in eval space
        y_pred.extend(preds.tolist())  # also in eval space now

    # --- confusion matrix
    norm = None if args.normalize == "none" else args.normalize  # 'true'|'pred'|'all' or None
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(eval_classes))), normalize=norm)

    # --- save CSV
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + eval_classes)
        for i, row in enumerate(cm):
            if norm:
                w.writerow([eval_classes[i]] + [f"{v:.4f}" for v in row])
            else:
                w.writerow([eval_classes[i]] + [int(v) for v in row])
    print(f"Saved confusion matrix to: {out_csv}")

    # --- simple per-class accuracy (row-wise recall)
    import numpy as np
    cm_np = np.array(cm, dtype=float)
    if not norm or norm == "pred":
        # compute recall from raw counts
        row_sums = cm_np.sum(axis=1, keepdims=True) + 1e-12
        recalls = np.diag(cm_np) / row_sums.squeeze(1)
    else:
        # when normalize='true' or 'all', diagonal already represents recall fractions per row
        recalls = np.diag(cm_np)

    worst = np.argsort(recalls)[:min(10, len(recalls))]
    print("\nLowest per-class recall (top 10):")
    for i in worst:
        print(f"  {eval_classes[i]:20s}  recall={recalls[i]*100:5.1f}%  total={int(cm_np[i].sum())}")

    # --- optional heatmap figure
    if args.fig:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(max(8, len(eval_classes)*0.3), max(6, len(eval_classes)*0.3)))
            im = plt.imshow(cm_np if norm else cm_np.astype(int), aspect='auto')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(ticks=range(len(eval_classes)), labels=eval_classes, rotation=90)
            plt.yticks(ticks=range(len(eval_classes)), labels=eval_classes)
            plt.title(f"Confusion Matrix ({'normalized' if norm else 'counts'})")
            plt.tight_layout()
            fig_path = Path(args.fig)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"Saved heatmap figure to: {fig_path}")
        except Exception as e:
            print("Skipping heatmap:", e)


# ----------------------------
# CLI commands
# ----------------------------
def cmd_crop(args):
    crop_animals(args.src, args.dst, score_thr=args.score, pad=args.pad, batch_det=args.batch_det)
    print("Crops saved to:", args.dst)

def cmd_train(args):
    set_seed(args.seed)
    device = get_device()
    print("device:", device)

    # AMP settings (new)
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_scaler = amp_enabled and (amp_dtype is torch.float16)

    sizes = [int(s) for s in args.sizes.split(",")]
    epochs = [int(e) for e in args.epochs.split(",")]
    assert len(sizes) == len(epochs), "provide same number of sizes and epochs (e.g., --sizes 224,320 --epochs 6,8)"

    train_loader, val_loader, classes = make_loaders(args.data, img_size=sizes[0],
                                                     batch=args.batch, val_split=0.15,
                                                     num_workers=args.workers, balanced=True)

    model = build_model(args.backbone, num_classes=len(classes)).to(device)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_ckpt = Path(args.out)/"best.pt"

    for stage, (sz, ep) in enumerate(zip(sizes, epochs), 1):
        print(f"\n== Stage {stage}: size {sz}, epochs {ep} ==")
        # rebuild loaders for new size
        train_loader, val_loader, _ = make_loaders(args.data, img_size=sz, batch=args.batch,
                                                   val_split=0.15, num_workers=args.workers, balanced=True)
        stage_mix_p = 0.0 if stage == 1 else 0.3  # off at 224, on at 320
        stage_mix_alpha = 0.2
        stage_best = train_one_stage(model, (train_loader, val_loader), device,
                                     epochs=ep, lr=args.lr, weight_decay=args.weight_decay,
                                     label_smoothing=0.1, mix_p=stage_mix_p, mix_alpha=stage_mix_alpha, ckpt_path=best_ckpt,
                                     amp_enabled=amp_enabled, amp_dtype=amp_dtype, use_scaler=use_scaler, opt_name=args.opt, scheduler_kind=args.scheduler)
        del train_loader, val_loader
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    # save class names & backbone info
    meta = {"classes": classes, "backbone": args.backbone}
    with open(Path(args.out)/"meta.json","w") as f: json.dump(meta, f, indent=2)
    print("\nTraining done. Best checkpoint at:", best_ckpt)

@torch.no_grad()
def cmd_eval(args):
    device = get_device()

    # 1) Read global training class list from the first ckpt
    meta_path0 = Path(args.ckpt[0]).with_name("meta.json")
    assert meta_path0.exists(), f"meta.json not found next to {args.ckpt[0]}"
    meta0 = json.load(open(meta_path0))
    train_classes = meta0["classes"]
    num_train = len(train_classes)
    train_class_to_idx = {c:i for i,c in enumerate(train_classes)}

    # 2) Build eval dataset (subset of train classes)
    ds = datasets.ImageFolder(args.data, transform=make_transforms(320, train=False))
    eval_classes = ds.classes
    missing = [c for c in eval_classes if c not in train_class_to_idx]
    if missing:
        raise ValueError(f"Eval classes not in training list: {missing}")
    subset_to_train = torch.tensor([train_class_to_idx[c] for c in eval_classes], device=device)

    val_loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # 3) Load models
    models_list = []
    names = []
    for ck in args.ckpt:
        meta_path = Path(ck).with_name("meta.json")
        backbone = "effb0"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            backbone = meta.get("backbone", backbone)
            if meta.get("classes") and meta["classes"] != train_classes:
                print(f"Warning: class list in {ck} differs; using first checkpoint's ordering.")
        m = build_model(backbone, num_classes=num_train).to(device)
        state = torch.load(ck, map_location=device)
        m.load_state_dict(state["model"])
        m.eval()
        models_list.append(m); names.append(backbone)
        # quick model size print
        nparams = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"Loaded {backbone:16s} from {ck}  | trainable params: {nparams/1e6:.2f}M")

    amp_enabled = (device.type == "cuda")
    crit = nn.CrossEntropyLoss()

    # helper to run one pass and (optionally) return per-model accuracies
    def pass_eval(return_per_model_acc=False, weights=None):
        tot_loss=tot_acc=n=0
        if return_per_model_acc:
            corr = [0]*len(models_list); cnt = 0
        for xb, yb in tqdm(val_loader, desc="eval"):
            xb, yb = xb.to(device), yb.to(device)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                # collect logits per model (restricted to eval classes)
                logits_each = []
                for m in models_list:
                    logits = m(xb)
                    if args.tta:
                        logits = (logits + m(torch.flip(xb, dims=[3]))) / 2
                    logits_eval = logits.index_select(1, subset_to_train)
                    logits_each.append(logits_eval)

                # per-model acc (on-the-fly)
                if return_per_model_acc:
                    for i, logit in enumerate(logits_each):
                        preds = logit.argmax(1)
                        corr[i] += (preds == yb).sum().item()
                    cnt += yb.size(0)

                # ensemble combine
                if weights is None:
                    logits_agg = sum(logits_each) / len(logits_each)
                else:
                    logits_agg = 0
                    for w, logit in zip(weights, logits_each):
                        logits_agg = logits_agg + (w * logit)

                loss = crit(logits_agg, yb)
                bs = yb.size(0)
                tot_loss += loss.item()*bs
                tot_acc  += (logits_agg.argmax(1) == yb).float().sum().item()
                n += bs

        res = (tot_loss/n, tot_acc/n)
        if return_per_model_acc:
            accs = [c/cnt for c in corr]
            return res, accs
        return res

    # Decide weights
    weights = None
    if args.weights is not None:
        assert len(args.weights) == len(models_list), "--weights must match # of ckpts"
        s = sum(args.weights); weights = [w/s for w in args.weights]
        print("Using manual weights:", [f"{w:.3f}" for w in weights])

    elif args.ensemble == "boost":
        # quick first pass to measure per-model accuracy (using same TTA setting)
        (_, _), accs = pass_eval(return_per_model_acc=True)
        # convert acc -> weights
        gamma = float(args.boost_gamma)
        eps = 1e-6
        raw = [(a+eps)**gamma for a in accs]
        s = sum(raw); weights = [r/s for r in raw]
        print("Per-model acc:", [f"{a*100:.1f}%" for a in accs])
        print("Boost weights :", [f"{w:.3f}" for w in weights])

    # Final pass (avg or weighted)
    val_loss, val_acc = pass_eval(return_per_model_acc=False, weights=weights)
    print(f"\nEnsemble ({'avg' if weights is None else 'boosted'})  loss {val_loss:.4f}  acc {val_acc:.4f}")

@torch.no_grad()
def cmd_predict(args):
    """
    Predict breed for a single image using one or more checkpoints (ensemble).
    If --label is provided (must match a training class), also prints accuracy (0/1) and CE loss.
    Supports optional --weights for a weighted ensemble (e.g., from boosted eval).
    """
    device = get_device()
    print("device:", device)

    # --- read training class list from the first checkpoint's meta.json
    meta_path0 = Path(args.ckpt[0]).with_name("meta.json")
    assert meta_path0.exists(), f"meta.json not found next to {args.ckpt[0]}"
    meta0 = json.load(open(meta_path0))
    train_classes = meta0["classes"]
    train_class_to_idx = {c: i for i, c in enumerate(train_classes)}
    num_classes = len(train_classes)

    # --- load image and build val transform
    img_path = Path(args.img)
    assert img_path.exists(), f"Image not found: {img_path}"
    im = Image.open(img_path).convert("RGB")
    tfm = make_transforms(args.size, train=False)
    x = tfm(im).unsqueeze(0).to(device)  # (1,C,H,W)

    # --- AMP settings
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.float16

    # --- load models
    models_list = []
    for ck in args.ckpt:
        meta_path = Path(ck).with_name("meta.json")
        backbone = "effb0"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            backbone = meta.get("backbone", backbone)
            if meta.get("classes") and meta["classes"] != train_classes:
                print(f"Warning: class list in {ck} differs from the first checkpoint; using the first one's ordering.")
        m = build_model(backbone, num_classes=num_classes).to(device)
        state = torch.load(ck, map_location=device)
        m.load_state_dict(state["model"])
        m.eval()
        models_list.append(m)
        print(f"Loaded {backbone} from {ck}")

    # --- optional weights (e.g., from boosted eval). Normalize to sum=1.
    ens_weights = None
    if getattr(args, "weights", None):
        assert len(args.weights) == len(models_list), "--weights must match number of --ckpt models"
        s = sum(args.weights)
        ens_weights = [w / s for w in args.weights]
        print("Using ensemble weights:", [f"{w:.3f}" for w in ens_weights])

    # --- forward (ensemble + optional TTA flip)
    with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits_each = []
        for m in models_list:
            logits = m(x)
            if args.tta:
                logits = (logits + m(torch.flip(x, dims=[3]))) / 2
            logits_each.append(logits)

        if ens_weights is None:
            logits_agg = sum(logits_each) / len(logits_each)
        else:
            logits_agg = sum(w * l for w, l in zip(ens_weights, logits_each))

    probs = F.softmax(logits_agg, dim=1).squeeze(0)
    topk = min(args.topk, num_classes)
    confs, idxs = torch.topk(probs, k=topk)
    idxs = idxs.tolist(); confs = confs.tolist()

    print("\nTop predictions:")
    for r, (ci, p) in enumerate(zip(idxs, confs), 1):
        print(f"  {r}. {train_classes[ci]}  —  {p*100:.2f}%")

    if args.label:
        def norm(s): return s.strip().lower().replace(" ", "_").replace("-", "_")
        label_norm = norm(args.label)
        name_map = {norm(c): c for c in train_classes}
        if label_norm not in name_map:
            cand = ", ".join(sorted(train_classes)[:10]) + (" ..." if len(train_classes) > 10 else "")
            raise ValueError(f"Label '{args.label}' not found in training classes. Example classes: {cand}")
        gt_name = name_map[label_norm]
        gt_idx = train_class_to_idx[gt_name]
        crit = nn.CrossEntropyLoss()
        loss = crit(logits_agg, torch.tensor([gt_idx], device=device)).item()
        pred_idx = int(torch.argmax(probs).item())
        acc = 1 if pred_idx == gt_idx else 0
        print(f"\nGround truth: {gt_name}")
        print(f"Predicted   : {train_classes[pred_idx]}  ({probs[pred_idx]*100:.2f}%)")
        print(f"Accuracy    : {acc}  (1=correct, 0=incorrect)")
        print(f"Loss (CE)   : {loss:.4f}")

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Bovine breed pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("crop", help="Detect & crop animals into a new dataset")
    sp.add_argument("--batch-det", type=int, default=4)
    sp.add_argument("--src", required=True)
    sp.add_argument("--dst", required=True)
    sp.add_argument("--score", type=float, default=0.7)
    sp.add_argument("--pad", type=float, default=0.06)
    sp.set_defaults(func=cmd_crop)

    sp = sub.add_parser("train", help="Train transfer model with progressive resizing")
    sp.add_argument("--opt", choices=["adamw","sgd"], default="adamw")
    sp.add_argument("--scheduler", choices=["plateau","cosine"], default="plateau")
    sp.add_argument("--label-smoothing", type=float, default=0.1)

    sp.add_argument("--data", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--backbone",
    choices=["effb0","convnext_tiny","effv2_s","swin_t","vit_b16",
             "densenet121","mobilenet_v3_large","nasnetalarge","nasnetamobile","xception"],
    default="effb0")
    sp.add_argument("--sizes", default="224,320")
    sp.add_argument("--epochs", default="6,8")
    sp.add_argument("--batch", type=int, default=32)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--weight-decay", type=float, default=1e-4)
    sp.add_argument("--workers", type=int, default=2)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--focus-classes", nargs="+", default=[],
                help="class names to upweight in the sampler")
    sp.add_argument("--focus-factor", type=float, default=2.0,
                help="multiply sampler weight for these classes")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("eval", help="Evaluate checkpoints on a dataset (with optional TTA & ensemble)")
    sp.add_argument("--ensemble", choices=["avg","boost"], default="avg",
                help="avg: uniform logits mean; boost: weight models by per-model accuracy on eval")
    sp.add_argument("--weights", nargs="+", type=float, default=None,
                help="optional manual weights for models (same order as --ckpt)")
    sp.add_argument("--boost-gamma", type=float, default=2.0,
                help="exponent for boost weights from per-model accuracy")

    sp.add_argument("--data", required=True)
    sp.add_argument("--ckpt", nargs="+", required=True, help="one or more .pt files")
    sp.add_argument("--batch", type=int, default=64)
    sp.add_argument("--workers", type=int, default=2)
    sp.add_argument("--tta", action="store_true")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("predict", help="Predict a single image (ensemble-friendly).")
    sp.add_argument("--img", required=True, help="path to a single image")
    sp.add_argument("--ckpt", nargs="+", required=True, help="one or more .pt files")
    sp.add_argument("--size", type=int, default=320, help="inference image size")
    sp.add_argument("--tta", action="store_true", help="enable simple TTA (flip)")
    sp.add_argument("--topk", type=int, default=5, help="show top-k predictions")
    sp.add_argument("--weights", nargs="+", type=float, default=None,
                help="optional per-model weights (same order as --ckpt); will be normalized to sum=1")
    sp.add_argument("--label", type=str, default=None, help="optional ground-truth class name to compute loss/accuracy")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("confmat", help="Compute confusion matrix (ensemble-friendly).")
    sp.add_argument("--data", required=True, help="dataset root (ImageFolder)")
    sp.add_argument("--ckpt", nargs="+", required=True, help="one or more .pt files")
    sp.add_argument("--size", type=int, default=320)
    sp.add_argument("--batch", type=int, default=128)
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--tta", action="store_true")
    sp.add_argument("--normalize", choices=["none","true","pred","all"], default="none",
                    help="sklearn normalization mode for confusion_matrix")
    sp.add_argument("--out", default="runs/confusion_matrix.csv")
    sp.add_argument("--fig", default="", help="optional path to save a PNG heatmap")
    sp.set_defaults(func=cmd_confmat)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()