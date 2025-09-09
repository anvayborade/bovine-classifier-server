from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Tuple
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageOps
import io, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models as tvm
from torchvision.transforms.functional import InterpolationMode

# Optional: used only if a checkpoint is an Ultralytics YOLO classification ckpt
from ultralytics import YOLO
import timm  # kept as a fallback if a ckpt truly was trained with timm

# -------------------- CONFIG --------------------
MODELS: Dict[str, dict] = {
    "effb0_d1d3": {"path": "runs/effb0_d1d3/best.pt", "imgsz": 320, "backbone": "effb0"},
    "mnv3l":      {"path": "runs/mnv3l/best.pt",      "imgsz": 320, "backbone": "mobilenet_v3_large"},
    "effv2s":     {"path": "runs/effv2s/best.pt",     "imgsz": 320, "backbone": "efficientnet_v2_s"},
}
DEFAULT_KEY = "effb0_d1d3"
LABELS_PATH = "labels.txt"  # fallback if meta.json not present
# ------------------------------------------------

def _bb_normalize(s: str) -> str:
    s = s.strip().lower().replace('-', '_').replace(' ', '_')
    s = s.replace('mobilenet_v3', 'mobilenet_v3')
    s = s.replace('efficientnet_v2', 'efficientnet_v2')
    # collapse your alias
    if s.startswith('effb0'):
        s = 'efficientnet_b0'
    if s in {'mnv3l','mbv3l','mobilenetv3_large'}:
        s = 'mobilenet_v3_large'
    if s in {'effv2_s','efficientnetv2_s'}:
        s = 'efficientnet_v2_s'
    return s

# torchvision builders (preferred)
def build_torchvision(backbone: str, num_classes: int) -> nn.Module:
    key = _bb_normalize(backbone)
    if key == 'efficientnet_b0':
        m = tvm.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    if key == 'efficientnet_v2_s':
        m = tvm.efficientnet_v2_s(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    if key == 'mobilenet_v3_large':
        m = tvm.mobilenet_v3_large(weights=None)
        # final linear is classifier[-1]
        # (classifier = [Dropout, Linear, Dropout, Linear])
        lin_idx = -1
        in_f = m.classifier[lin_idx].in_features
        m.classifier[lin_idx] = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(f"Unknown torchvision backbone '{backbone}' (normalized '{key}')")

# timm fallback (rarely needed now)
def build_timm(backbone: str, num_classes: int) -> nn.Module:
    key = _bb_normalize(backbone)
    arch_map = {
        'efficientnet_b0': 'efficientnet_b0',
        'mobilenet_v3_large': 'mobilenetv3_large_100',
        'efficientnet_v2_s': 'efficientnetv2_s',
    }
    arch = arch_map.get(key, key)
    m = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    return m

def make_tfm(imgsz: int):
    resize_to = int(round(imgsz / 0.875))  # ImageNet eval: e.g., 320 -> 366
    return transforms.Compose([
        transforms.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

app = FastAPI(title="Bovine Breed API (TorchVision + Logit Ensemble)")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    model: str
    label: str
    confidence: float
    top5: List[Tuple[str, float]]

# ---------- helpers: labels / meta ----------
@lru_cache(maxsize=1)
def labels_fallback() -> List[str]:
    p = Path(LABELS_PATH)
    if not p.exists():
        raise FileNotFoundError(f"labels.txt not found at {p.resolve()}. Create one label per line.")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def read_meta_for_ckpt(ckpt_path: str):
    meta_path = Path(ckpt_path).with_name("meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

# ---------- model wrapper ----------
class Wrapped:
    def __init__(self, kind: str, obj, imgsz: int, names: List[str], tfm=None):
        self.kind = kind        # "torchvision", "ultralytics", or "timm"
        self.obj = obj
        self.imgsz = imgsz
        self.names = names
        self.tfm = tfm

def _load_state_dict_smart(net: nn.Module, ckpt):
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    cleaned = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (up to 5): {unexpected[:5]}")

@lru_cache(maxsize=len(MODELS))
def load_model(key: str) -> Wrapped:
    info = MODELS[key]
    ck = info["path"]
    imgsz = info.get("imgsz", 224)

    meta = read_meta_for_ckpt(ck)
    backbone = meta.get("backbone", info.get("backbone", "efficientnet_b0"))
    classes_from_meta = meta.get("classes", None)
    names = classes_from_meta or labels_fallback()

    # 1) Try Ultralytics format first
    try:
        m = YOLO(ck)
        ul_names = getattr(m.model, "names", None) or getattr(m, "names", None)
        if isinstance(ul_names, dict):
            ul_names = [ul_names[i] for i in range(len(ul_names))]
        return Wrapped("ultralytics", m, imgsz, ul_names or names)
    except Exception:
        pass

    # 2) Prefer TorchVision (matches your checkpoints)
    try:
        net = build_torchvision(backbone, num_classes=len(names))
        ckpt = torch.load(ck, map_location="cpu")
        _load_state_dict_smart(net, ckpt)
        net.eval()
        tfm = make_tfm(imgsz)
        return Wrapped("torchvision", net, imgsz, names, tfm=tfm)
    except Exception as e_tv:
        print(f"[INFO] TorchVision load failed for '{key}' with backbone '{backbone}': {e_tv}")

    # 3) Fallback to timm if truly needed
    net = build_timm(backbone, num_classes=len(names))
    ckpt = torch.load(ck, map_location="cpu")
    _load_state_dict_smart(net, ckpt)
    net.eval()
    tfm = make_tfm(imgsz)
    return Wrapped("timm", net, imgsz, names, tfm=tfm)

@lru_cache(maxsize=1)
def canonical_labels() -> List[str]:
    first = load_model(DEFAULT_KEY)
    return first.names

# ---------- inference ----------
@torch.inference_mode()
def logits_from_wrapped(w: Wrapped, image: Image.Image, tta: bool) -> torch.Tensor:
    if w.kind in {"torchvision", "timm"}:
        x = w.tfm(image).unsqueeze(0)
        logits = w.obj(x)
        if tta:
            xh = torch.flip(x, dims=[3])  # horizontal flip
            logits = (logits + w.obj(xh)) / 2
        return logits
    else:
        # Ultralytics returns probs; convert to pseudo-logits
        res = w.obj.predict(image, imgsz=w.imgsz, verbose=False)[0]
        p = res.probs.data.float().unsqueeze(0)
        if tta:
            res2 = w.obj.predict(ImageOps.mirror(image), imgsz=w.imgsz, verbose=False)[0]
            p = (p + res2.probs.data.float().unsqueeze(0)) / 2.0
        eps = 1e-9
        return torch.log(p + eps)

def align_logits_to_canonical(w: Wrapped, logits: torch.Tensor, canon: List[str]) -> torch.Tensor:
    if w.names == canon:
        return logits
    idx_map = {name: i for i, name in enumerate(w.names)}
    cols = [idx_map[name] for name in canon]
    return logits[:, cols]

# ================= API =================
app = FastAPI(title="Bovine Breed API (TorchVision + Logit Ensemble)")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    model: str
    label: str
    confidence: float
    top5: List[Tuple[str, float]]

@app.get("/models")
def list_models():
    return {k: k for k in MODELS} | {"ensemble": "ensemble"}

@app.get("/labels")
def labels(model: str = Query(DEFAULT_KEY)):
    if model == "ensemble":
        return {"model": "ensemble", "labels": canonical_labels()}
    w = load_model(model)
    return {"model": model, "labels": w.names}

@app.post("/predict", response_model=PredictResponse)
async def predict_single(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_KEY),
    tta: bool = Query(False),
    gamma: float = Query(1.0)
):
    if model not in MODELS:
        model = DEFAULT_KEY
    w = load_model(model)

    img = Image.open(io.BytesIO(await file.read()))
    img = ImageOps.exif_transpose(img).convert("RGB")

    logits = logits_from_wrapped(w, img, tta=tta)
    probs  = torch.softmax(logits, dim=1)
    if gamma and gamma != 1.0:
        probs = probs.pow(gamma)
        probs = probs / probs.sum(dim=1, keepdim=True)

    p = probs[0].cpu().numpy()
    names = w.names
    top1 = int(np.argmax(p))
    top5 = np.argsort(p)[::-1][:5]
    return PredictResponse(
        model=model,
        label=names[top1],
        confidence=float(p[top1]),
        top5=[(names[i], float(p[i])) for i in top5],
    )

@app.post("/predict_ensemble", response_model=PredictResponse)
async def predict_ensemble(
    file: UploadFile = File(...),
    weights: str = Query("", description="e.g. effb0_d1d3:0.294,mnv3l:0.341,effv2s:0.365"),
    tta: bool = Query(False),
    gamma: float = Query(1.0)
):
    # parse/normalize weights
    wmap: Dict[str, float] = {}
    if weights:
        for part in weights.split(","):
            k, v = part.split(":")
            if k.strip() in MODELS:
                wmap[k.strip()] = float(v)
    if not wmap:
        wmap = {k: 1.0 for k in MODELS}
    s = sum(wmap.values())
    wmap = {k: v/s for k, v in wmap.items()}

    img = Image.open(io.BytesIO(await file.read()))
    img = ImageOps.exif_transpose(img).convert("RGB")
    canon = canonical_labels()

    logits_agg = None
    for k in MODELS:
        wk = load_model(k)
        l = logits_from_wrapped(wk, img, tta=tta)
        l = align_logits_to_canonical(wk, l, canon)
        logits_agg = l * wmap[k] if logits_agg is None else logits_agg + l * wmap[k]

    probs = torch.softmax(logits_agg, dim=1)
    if gamma and gamma != 1.0:
        probs = probs.pow(gamma)
        probs = probs / probs.sum(dim=1, keepdim=True)

    p = probs[0].cpu().numpy()
    top1 = int(np.argmax(p))
    top5 = np.argsort(p)[::-1][:5]
    return PredictResponse(
        model="ensemble",
        label=canon[top1],
        confidence=float(p[top1]),
        top5=[(canon[i], float(p[i])) for i in top5],
    )

@app.get("/ping")
def ping():
    return {"ok": True}