import argparse, json, re
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.amp import autocast

# reuse your helpers
from bovine_pipeline import build_model, make_transforms, get_device

def norm_name(s: str) -> str:
    s = s.strip().lower().replace("_"," ").replace("-"," ")
    s = re.sub(r"\s+"," ",s).strip()
    return s

@torch.no_grad()
def extract_feature_single(img_path, ckpts, size=320, tta=False):
    """
    Returns:
      feat_cat: np.ndarray of shape (1, D_total)
      train_classes: list[str] taken from the first ckpt's meta.json
    """
    device = get_device()
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float16

    # image -> tensor
    im = Image.open(img_path).convert("RGB")
    tfm = make_transforms(size, train=False)
    x = tfm(im).unsqueeze(0).to(device)

    # class list from first ckpt
    meta0 = json.load(open(Path(ckpts[0]).with_name("meta.json"), "r"))
    train_classes = meta0["classes"]; num_train = len(train_classes)

    feats = []
    for ck in ckpts:
        meta = json.load(open(Path(ck).with_name("meta.json"), "r"))
        backbone = meta.get("backbone", "effb0")
        m = build_model(backbone, num_train).to(device)
        state = torch.load(ck, map_location=device)
        m.load_state_dict(state["model"]); m.eval()

        # grab input to final Linear (penultimate features)
        # pick the right head per backbone
        if backbone == "effb0":
            head = m.classifier[1]
        elif backbone == "convnext_tiny":
            head = m.classifier[2]
        elif backbone == "mobilenet_v3_large":
            head = m.classifier[3]
        elif backbone == "effv2_s":
            head = m.classifier[1]
        elif backbone == "swin_t":
            head = m.head
        elif backbone == "vit_b16":
            head = m.heads.head
        elif backbone == "densenet121":
            head = m.classifier
        else:
            raise ValueError(f"Unsupported backbone for single-image extraction: {backbone}")

        buf = {"x": None}
        def _hook(module, inputs, output): buf["x"] = inputs[0].detach()
        h = head.register_forward_hook(_hook)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _ = m(x)
            f = buf["x"]  # (1, F)
            if tta:
                _ = m(torch.flip(x, dims=[3]))
                f2 = buf["x"]
                f = 0.5*(f + f2)
        feats.append(f.float().cpu())
        h.remove()

    feat_cat = torch.cat(feats, dim=1).numpy()  # (1, sum(F_i))
    return feat_cat, train_classes, x  # return x too for optional neural fusion

@torch.no_grad()
def neural_probs_single(x, ckpts, train_classes, tta=False, weights=None):
    """Return neural ensemble probabilities (1, C_train) for the same image tensor x."""
    device = x.device
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float16
    C = len(train_classes)

    # normalize weights
    if weights is not None:
        w = np.array(weights, dtype=np.float64); w = (w / w.sum()).tolist()
    else:
        w = [1.0/len(ckpts)] * len(ckpts)

    logits_agg = None
    for wi, ck in zip(w, ckpts):
        meta = json.load(open(Path(ck).with_name("meta.json"), "r"))
        backbone = meta.get("backbone", "effb0")
        m = build_model(backbone, C).to(device)
        state = torch.load(ck, map_location=device)
        m.load_state_dict(state["model"]); m.eval()

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            lo = m(x)
            if tta:
                lo = 0.5*(lo + m(torch.flip(x, dims=[3])))

        logits_agg = lo*wi if logits_agg is None else (logits_agg + lo*wi)

    p = F.softmax(logits_agg, dim=1).cpu().numpy()  # (1, C)
    return p

def main():
    ap = argparse.ArgumentParser("Single-image prediction using calibrated SVM (+optional fusion with neural ensemble)")
    ap.add_argument("--svm-in", required=True, help="joblib file saved by svm_fusion.py (contains clf + optional PCA + classes)")
    ap.add_argument("--img", required=True, help="path to image")
    ap.add_argument("--ckpt", nargs="+", required=True, help="same checkpoints used to train the SVM features (same order)")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--topk", type=int, default=5)

    # fusion
    ap.add_argument("--alpha", type=float, default=1.0, help="fusion: p=(1-a)*neural + a*svm ; use 1.0 for SVM-only")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="optional neural ensemble weights (same order as --ckpt)")

    # optional label to see CE loss/accuracy
    ap.add_argument("--label", nargs="+", default=None, help="optional ground-truth aliases to compute CE & accuracy")
    args = ap.parse_args()

    # load SVM (+ PCA) and class list
    import joblib
    bundle = joblib.load(args.svm_in)
    clf = bundle["clf"]
    pca = bundle.get("pca", None)
    train_classes = bundle["classes"]
    C = len(train_classes)

    # features from image
    feat, train_classes_ck, x = extract_feature_single(args.img, args.ckpt, size=args.size, tta=args.tta)
    assert train_classes_ck == train_classes, "Class list mismatch: use the SAME ckpts/order as used to train the SVM."

    if pca is not None:
        feat = pca.transform(feat)

    # SVM probabilities
    svm_proba = clf.predict_proba(feat)  # (1, C)

    # Optional fusion with neural ensemble
    if args.alpha < 1.0:
        p_neural = neural_probs_single(x, args.ckpt, train_classes, tta=args.tta, weights=args.weights)  # (1, C)
        p_final = (1.0 - args.alpha) * p_neural + args.alpha * svm_proba
        used = f"FUSED (alpha={args.alpha:.2f})"
    else:
        p_final = svm_proba
        used = "SVM-only"

    # Top-k
    probs = p_final[0]
    idxs = np.argsort(-probs)[:max(1, min(args.topk, C))]
    print(f"\nTop predictions [{used}]:")
    for r, i in enumerate(idxs, 1):
        print(f"  {r}. {train_classes[i]}  —  {probs[i]*100:.2f}%")

    # Optional CE loss/accuracy if label provided
    if args.label:
        aliases = list(args.label)
        if len(aliases) > 1:
            aliases.append(" ".join(args.label))
        STOP = {"cattle","cow","buffalo","breed","bull","heifer","dairy"}
        def norm(s): 
            s = s.lower().replace("_"," ").replace("-"," ")
            toks = [t for t in re.findall(r"[a-z0-9]+", s)]
            toks = [t for t in toks if t not in STOP]
            return "".join(toks)
        import re
        canon = {norm(c): c for c in train_classes}
        gt_name = None
        tried = []
        for a in aliases:
            tried.append(a)
            k = norm(a)
            if k in canon:
                gt_name = canon[k]; break
        if gt_name is None:
            print(f"\n(Label note) Could not map any of {tried} to a class.")
        else:
            gt_idx = train_classes.index(gt_name)
            # CE = -log p_true
            p_true = float(probs[gt_idx])
            ce = -np.log(max(p_true, 1e-12))
            pred_idx = int(idxs[0])
            acc = int(pred_idx == gt_idx)
            print(f"\nGround truth: {gt_name}")
            print(f"Predicted   : {train_classes[pred_idx]}  ({probs[pred_idx]*100:.2f}%)")
            print(f"Accuracy    : {acc}  (1=correct, 0=incorrect)")
            print(f"Loss (CE)   : {ce:.4f}")

if __name__ == "__main__":
    main()