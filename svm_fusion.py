# svm_fusion.py
import argparse, json, re
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.amp import autocast

# Reuse your helpers
from bovine_pipeline import build_model, make_transforms, get_device

# sklearn bits
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

def norm_name(s: str) -> str:
    s = s.strip().lower().replace("_"," ").replace("-"," ")
    s = re.sub(r"\s+"," ",s).strip()
    return s

def subset_index_mapping(eval_classes, train_classes):
    tmap = {c:i for i,c in enumerate(train_classes)}
    miss = [c for c in eval_classes if c not in tmap]
    if miss:
        raise ValueError(f"Eval classes not in training classes: {miss[:8]}{'...' if len(miss)>8 else ''}")
    return torch.tensor([tmap[c] for c in eval_classes], dtype=torch.long)

def dataloader(root, size, batch, workers):
    tfm = make_transforms(size, train=False)
    ds  = datasets.ImageFolder(root, transform=tfm)
    pin = torch.cuda.is_available()
    return ds, DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin)

def find_head_module(model, backbone: str):
    if backbone == "effb0":
        return model.classifier[1]
    if backbone == "convnext_tiny":
        return model.classifier[2]
    if backbone == "mobilenet_v3_large":
        return model.classifier[3]
    if backbone == "effv2_s":
        return model.classifier[1]
    if backbone == "swin_t":
        return model.head
    if backbone == "vit_b16":
        return model.heads.head
    if backbone == "densenet121":
        return model.classifier
    raise ValueError(f"Head finder not defined for backbone={backbone}")

@torch.no_grad()
def extract_features(root, ckpts, size=320, batch=64, workers=4, tta=False):
    """
    Returns:
      feats_concat: np.ndarray [N, D_total]  (concat features from all models)
      labels: np.ndarray [N]
      train_classes: list[str]
      eval_classes: list[str]
      per_model_dims: list[int]
    """
    device = get_device()
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float16

    meta0 = json.load(open(Path(ckpts[0]).with_name("meta.json"), "r"))
    train_classes = meta0["classes"]; num_train = len(train_classes)

    ds, loader = dataloader(root, size, batch, workers)
    eval_classes = ds.classes

    models, hooks, dims = [], [], []
    for ck in ckpts:
        meta = json.load(open(Path(ck).with_name("meta.json"), "r"))
        backbone = meta.get("backbone", "effb0")
        m = build_model(backbone, num_train).to(device)
        state = torch.load(ck, map_location=device)
        m.load_state_dict(state["model"]); m.eval()
        head = find_head_module(m, backbone)

        buf = {"x": None}
        def _hook(module, inputs, output):
            buf["x"] = inputs[0].detach()   # penultimate features (B,F)
        h = head.register_forward_hook(_hook)

        models.append((m, buf, backbone))
        hooks.append(h)
        dims.append(None)

    all_feats, all_labels = [], []
    for xb, yb in tqdm(loader, desc=f"extract[{Path(root).name}]"):
        xb = xb.to(device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            feats_stack = []
            for i, (m, buf, _b) in enumerate(models):
                _ = m(xb)                  # trigger hook
                f = buf["x"]
                if tta:
                    _ = m(torch.flip(xb, dims=[3]))
                    f2 = buf["x"]
                    f = 0.5*(f + f2)
                if dims[i] is None: dims[i] = f.shape[1]
                feats_stack.append(f.float().cpu())
        feats_cat = torch.cat(feats_stack, dim=1)     # (B, sum(F_i))
        all_feats.append(feats_cat)
        all_labels.append(yb)

    feats_concat = torch.cat(all_feats, dim=0).numpy()
    labels_np    = torch.cat(all_labels, dim=0).numpy()
    for h in hooks: h.remove()
    return feats_concat, labels_np, train_classes, eval_classes, [d for d in dims]

@torch.no_grad()
def per_model_probs_and_acc(root, ckpts, size=320, batch=64, workers=4, tta=False):
    """
    For each model, compute probs on eval set and its top-1 accuracy.
    Returns: list[np.ndarray (N,C_eval)], eval_classes, labels (np.ndarray), acc_list
    """
    device = get_device()
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float16

    meta0 = json.load(open(Path(ckpts[0]).with_name("meta.json"), "r"))
    train_classes = meta0["classes"]; num_train = len(train_classes)

    ds, loader = dataloader(root, size, batch, workers)
    eval_classes = ds.classes
    labels = np.array([y for _, y in ds.samples], dtype=np.int64)  # order matches loader (shuffle=False)
    subset_map = subset_index_mapping(eval_classes, train_classes).to(device)

    probs_list, accs = [], []
    for ck in ckpts:
        meta = json.load(open(Path(ck).with_name("meta.json"), "r"))
        backbone = meta.get("backbone", "effb0")
        m = build_model(backbone, num_train).to(device)
        state = torch.load(ck, map_location=device)
        m.load_state_dict(state["model"]); m.eval()

        batch_probs = []
        for xb, _ in tqdm(loader, desc=f"probs[{Path(root).name}|{backbone}]"):
            xb = xb.to(device)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                lo = m(xb)
                if tta:
                    lo = 0.5*(lo + m(torch.flip(xb, dims=[3])))
                lo_eval = lo.index_select(1, subset_map)   # align columns to eval order
                pb = F.softmax(lo_eval, dim=1)
            batch_probs.append(pb.cpu())
        p = torch.cat(batch_probs, dim=0).numpy()         # (N, C_eval)
        preds = p.argmax(1)
        acc = (preds == labels).mean().item()
        probs_list.append(p)
        accs.append(acc)
    return probs_list, eval_classes, labels, accs

def compute_boost_weights(accs, gamma=2.0):
    """w_i ∝ (acc_i ** gamma), normalized to sum=1."""
    a = np.array(accs, dtype=np.float64)
    a = np.maximum(a, 1e-6)
    w = a**float(gamma)
    w = w / w.sum()
    return w.tolist()

def main():
    p = argparse.ArgumentParser("SVM on deep features + boosted neural fusion")
    p.add_argument("--data-train", required=True, help="ImageFolder root for SVM training")
    p.add_argument("--data-val",   required=True, help="ImageFolder root for evaluation")
    p.add_argument("--ckpt", nargs="+", required=True, help="one or more .pt checkpoints")
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tta", action="store_true", help="flip TTA for features and neural probs")

    # PCA
    p.add_argument("--pca", type=int, default=512, help="PCA dims (0 disables)")
    p.add_argument("--pca-whiten", action="store_true", help="use whitening (recommended)")
    # SVM / calibration
    p.add_argument("--C", type=float, default=0.5, help="LinearSVC C (0.5 often converges faster)")
    p.add_argument("--cv", type=int, default=2, help="CV folds for calibration (2 is faster)")
    p.add_argument("--svm-out", default="runs/svm_calibrated.joblib", help="where to save calibrated SVM (+PCA)")

    # Neural fusion
    p.add_argument("--alpha", type=float, default=0.3, help="weight for SVM probs in fusion: p=(1-a)*neural + a*svm")
    p.add_argument("--weights", nargs="+", type=float, default=None, help="manual neural ensemble weights (sum auto-normalized)")
    p.add_argument("--boost", action="store_true", help="compute boosted weights from per-model val accuracy")
    p.add_argument("--boost-gamma", type=float, default=2.0, help="gamma for boost weighting (w∝acc^gamma)")
    args = p.parse_args()

    # 1) Feature extraction
    Xtr, ytr, train_classes, _, _ = extract_features(args.data_train, args.ckpt, size=args.size, batch=args.batch, workers=args.workers, tta=args.tta)
    Xva, yva, train_classes2, eval_classes, _ = extract_features(args.data_val,   args.ckpt, size=args.size, batch=args.batch, workers=args.workers, tta=args.tta)
    assert train_classes2 == train_classes, "Class ordering mismatch across checkpoints"

    # 2) PCA (optional, big speedup)
    if args.pca and args.pca > 0 and args.pca < Xtr.shape[1]:
        print(f"\nPCA → {args.pca} dims (randomized, whiten={args.pca_whiten})")
        pca = PCA(n_components=int(args.pca), svd_solver="randomized", whiten=args.pca_whiten, random_state=42)
        Xtr = pca.fit_transform(Xtr)
        Xva = pca.transform(Xva)
        pca_obj = pca
        print("Shapes after PCA:", Xtr.shape, Xva.shape)
    else:
        pca_obj = None
        print("\nPCA skipped.")

    # 3) Fit calibrated LinearSVC (fast config)
    n, d = Xtr.shape
    use_dual = not (n > d)  # liblinear tip: dual=False when n_samples > n_features
    print(f"\nFitting LinearSVC (C={args.C}, dual={use_dual}, tol=1e-3, max_iter=2000) + calibration (cv={args.cv})")
    base = LinearSVC(C=args.C, dual=use_dual, tol=1e-3, max_iter=2000, verbose=1)
    clf  = CalibratedClassifierCV(base, method="sigmoid", cv=args.cv, n_jobs=-1)
    clf.fit(Xtr, ytr)

    # 4) Evaluate SVM alone on eval
    svm_proba = clf.predict_proba(Xva)  # (N, C_train)
    tmap = {c:i for i,c in enumerate(train_classes)}
    subset_cols = np.array([tmap[c] for c in eval_classes], dtype=int)
    svm_proba_eval = svm_proba[:, subset_cols]
    svm_pred = svm_proba_eval.argmax(1)
    svm_acc  = accuracy_score(yva, svm_pred)
    print(f"\nSVM (calibrated) accuracy on eval: {svm_acc:.4f}")

    # 5) Neural ensemble: probs (uniform / manual / boosted)
    #    If manual weights provided -> use those.
    #    Else if --boost -> compute per-model acc on eval and set weights ∝ acc^gamma.
    if args.weights is not None:
        w = np.array(args.weights, dtype=np.float64)
        w = (w / w.sum()).tolist()
        print("Using manual ensemble weights:", [f"{x:.3f}" for x in w])
        manual_weights = w
        boosted = False
    else:
        manual_weights = None
        boosted = args.boost

    # Get per-model probs and accs
    probs_list, eval_classes2, labels_eval, accs = per_model_probs_and_acc(args.data_val, args.ckpt,
                                                                           size=args.size, batch=args.batch,
                                                                           workers=args.workers, tta=args.tta)
    assert eval_classes2 == eval_classes
    if manual_weights is not None:
        weights = manual_weights
    elif boosted:
        weights = compute_boost_weights(accs, gamma=args.boost_gamma)
        print("Boost weights (acc^gamma):", [f"{x:.3f}" for x in weights], " | per-model acc:", [f"{a:.3f}" for a in accs])
    else:
        weights = [1.0/len(probs_list)] * len(probs_list)
        print("Uniform ensemble weights:", [f"{x:.3f}" for x in weights], " | per-model acc:", [f"{a:.3f}" for a in accs])

    # Fuse neural probs with those weights
    neural_p = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        neural_p += w * p
    neural_pred = neural_p.argmax(1)
    neural_acc  = accuracy_score(yva, neural_pred)
    print(f"Neural ensemble accuracy on eval: {neural_acc:.4f}")

    # 6) Fusion: p = (1 - alpha) * neural + alpha * svm
    a = float(args.alpha)
    p_fused = (1.0 - a) * neural_p + a * svm_proba_eval
    fused_pred = p_fused.argmax(1)
    fused_acc  = accuracy_score(yva, fused_pred)
    print(f"Fused (alpha={a:.2f}) accuracy on eval: {fused_acc:.4f}")

    # 7) Per-class report (fused)
    print("\nPer-class report (fused):")
    print(classification_report(yva, fused_pred, target_names=eval_classes, digits=3))

    # 8) Save SVM (+ PCA pipeline if used)
    if args.svm_out:
        try:
            import joblib
            Path(args.svm_out).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"clf": clf, "pca": pca_obj, "classes": train_classes}, args.svm_out)
            print(f"\nSaved calibrated SVM (and PCA if any) to: {args.svm_out}")
        except Exception as e:
            print(f"Could not save SVM: {e}")

if __name__ == "__main__":
    main()
