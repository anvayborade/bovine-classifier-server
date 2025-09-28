# server.py
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Literal
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageOps
import io, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models as tvm
from torchvision.transforms.functional import InterpolationMode

# Optional: used only if a checkpoint is an Ultralytics YOLO classification ckpt
from ultralytics import YOLO
import timm  # fallback if a ckpt truly was trained with timm

import os
import httpx
import google.generativeai as genai

from pydantic import BaseModel
from typing import List

import asyncio, time, math
from typing import List, Optional, Dict, Literal
from fastapi import HTTPException
import math, httpx

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
        lin_idx = -1
        in_f = m.classifier[lin_idx].in_features
        m.classifier[lin_idx] = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(f"Unknown torchvision backbone '{backbone}' (normalized '{key}')")

# timm fallback
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

# =============== SINGLE APP + CORS ===============
app = FastAPI(title="Bovine Breed API (TorchVision + Logit Ensemble)")
app.add_middleware(
    CORSMiddleware,
    # For dev you can keep this permissive. If you later use cookies, switch to allow_origins=[].
    allow_origin_regex=r".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============== SCHEMAS =========================
class PredictResponse(BaseModel):
    model: str
    label: str
    confidence: float
    top5: List[Tuple[str, float]]

class Place(BaseModel):
    id: int
    name: Optional[str] = None
    lat: float
    lon: float
    category: str
    tags: Dict[str, str] = {}

class AiReq(BaseModel):
    breed: str
    location: Optional[str] = None
    extra: Optional[str] = None

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

    # 2) Prefer TorchVision
    try:
        net = build_torchvision(backbone, num_classes=len(names))
        ckpt = torch.load(ck, map_location="cpu")
        _load_state_dict_smart(net, ckpt)
        net.eval()
        tfm = make_tfm(imgsz)
        return Wrapped("torchvision", net, imgsz, names, tfm=tfm)
    except Exception as e_tv:
        print(f"[INFO] TorchVision load failed for '{key}' with backbone '{backbone}': {e_tv}")

    # 3) Fallback to timm
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

# ================= ROUTES =================
@app.get("/healthz")
def healthz():
    return {"ok": True}

#class PlacesResponse(BaseModel):
#    places: List[Place]

_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
_HTTP_TIMEOUT = httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0)  # per mirror
_PLACES_CACHE: dict = {}  # key -> (ts, list[Place])
_PLACES_TTL_SEC = 300     # 5 minutes

def _bbox_from_center(lat: float, lon: float, radius_m: int):
    radius_km = radius_m / 1000.0
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.01, math.cos(math.radians(lat))))
    left = lon - dlon
    right = lon + dlon
    top = lat + dlat
    bottom = lat - dlat
    return left, top, right, bottom

def _places_cache_key(lat: float, lon: float, radius_m: int, type_: str, limit: int):
    # round lat/lon to ~100m and bucket radius to reduce cache keys
    return (round(lat, 3), round(lon, 3), int((radius_m // 1000) * 1000), type_, limit)

async def _overpass_first_ok(query: str):
    async with httpx.AsyncClient(
        headers={"User-Agent": "BovineApp/1.0 (contact: dev@yourapp.example)"},
        timeout=_HTTP_TIMEOUT
    ) as client:
        async def call(url):
            try:
                r = await client.post(url, data={"data": query})
                if r.status_code == 200:
                    return r.json()
            except Exception:
                return None
            return None
        tasks = [asyncio.create_task(call(u)) for u in _OVERPASS_MIRRORS]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=10)
        for p in pending:
            p.cancel()
        for d in done:
            js = d.result()
            if js:
                return js
        return None

@app.get("/places", response_model=list[Place])
async def places(
    lat: float,
    lon: float,
    radius_m: int = 50000,
    type: Literal["vet", "market", "dairy"] = "vet",
    limit: int = 30,
    nocache: bool = False,  # <-- new
):
    key = _places_cache_key(lat, lon, radius_m, type, limit)
    now = time.time()

    # serve from cache ONLY if not forced and cached payload is non-empty
    if not nocache and key in _PLACES_CACHE:
        ts, payload = _PLACES_CACHE[key]
        if now - ts < _PLACES_TTL_SEC and payload:
            print(f"[places] cache hit {type} -> {len(payload)}")
            return payload

    # -------- build and run Overpass query (unchanged patterns) ----------
    # inside /places(...)
    if type == "vet":
        filters = f"""
      node["amenity"="veterinary"](around:{radius_m},{lat},{lon});
      way["amenity"="veterinary"](around:{radius_m},{lat},{lon});
      relation["amenity"="veterinary"](around:{radius_m},{lat},{lon});

      node["healthcare"="veterinary"](around:{radius_m},{lat},{lon});
      way["healthcare"="veterinary"](around:{radius_m},{lat},{lon});
      relation["healthcare"="veterinary"](around:{radius_m},{lat},{lon});

      node["shop"="veterinary"](around:{radius_m},{lat},{lon});
      way["shop"="veterinary"](around:{radius_m},{lat},{lon});
      relation["shop"="veterinary"](around:{radius_m},{lat},{lon});

      node["shop"="pet"]["veterinary"="yes"](around:{radius_m},{lat},{lon});
      way["shop"="pet"]["veterinary"="yes"](around:{radius_m},{lat},{lon});
      relation["shop"="pet"]["veterinary"="yes"](around:{radius_m},{lat},{lon});

      node["amenity"="clinic"]["name"~"(?i)(vet|animal|pet)"](around:{radius_m},{lat},{lon});
      way["amenity"="clinic"]["name"~"(?i)(vet|animal|pet)"](around:{radius_m},{lat},{lon});
      relation["amenity"="clinic"]["name"~"(?i)(vet|animal|pet)"](around:{radius_m},{lat},{lon});
    """
    elif type == "market":
        filters = f"""
      node["amenity"="marketplace"](around:{radius_m},{lat},{lon});
      way["amenity"="marketplace"](around:{radius_m},{lat},{lon});
      relation["amenity"="marketplace"](around:{radius_m},{lat},{lon});

      node[~"name"~"(?i)(mandi|market)"](around:{radius_m},{lat},{lon});
      way[~"name"~"(?i)(mandi|market)"](around:{radius_m},{lat},{lon});
      relation[~"name"~"(?i)(mandi|market)"](around:{radius_m},{lat},{lon});
    """
    else:  # type == "dairy"
    # dairy plants / collection / brands (Amul, Mother Dairy, Country Delight, Nandini, Hatsun, Verka, etc.)
        filters = f"""
      node["industrial"="dairy"](around:{radius_m},{lat},{lon});
      way["industrial"="dairy"](around:{radius_m},{lat},{lon});
      relation["industrial"="dairy"](around:{radius_m},{lat},{lon});

      node["shop"="dairy"](around:{radius_m},{lat},{lon});
      way["shop"="dairy"](around:{radius_m},{lat},{lon});
      relation["shop"="dairy"](around:{radius_m},{lat},{lon});

      node[~"name"~"(?i)(dairy|milk|collection|amul|mother dairy|country delight|nandini|hatsun|verka|aavin|parag|gokul|heritage|milky mist)"](around:{radius_m},{lat},{lon});
      way[~"name"~"(?i)(dairy|milk|collection|amul|mother dairy|country delight|nandini|hatsun|verka|aavin|parag|gokul|heritage|milky mist)"](around:{radius_m},{lat},{lon});
      relation[~"name"~"(?i)(dairy|milk|collection|amul|mother dairy|country delight|nandini|hatsun|verka|aavin|parag|gokul|heritage|milky mist)"](around:{radius_m},{lat},{lon});
    """


    overpass_query = f"""
    [out:json][timeout:12];
    (
      {filters}
    );
    out center tags;
    """

    elements = []
    try:
        js = await _overpass_first_ok(overpass_query)
        if js:
            elements = js.get("elements", [])
    except Exception as e:
        print(f"[places] overpass error: {e}")

    payload: list[dict] = []
    for el in elements:
        latv = el.get("lat") or (el.get("center") or {}).get("lat")
        lonv = el.get("lon") or (el.get("center") or {}).get("lon")
        if latv is None or lonv is None:
            continue
        tags = el.get("tags", {}) or {}
        payload.append({
            "id": int(el["id"]),
            "name": tags.get("name"),
            "lat": float(latv),
            "lon": float(lonv),
            "category": type,
            "tags": {k: str(v) for k, v in tags.items()},
        })
        if len(payload) >= limit:
            break

    # Nominatim fallback only if still empty
    if not payload:
        try:
            left, top, right, bottom = _bbox_from_center(lat, lon, radius_m)
            async with httpx.AsyncClient(
                headers={"User-Agent": "BovineApp/1.0 (contact: dev@yourapp.example)"},
                timeout=_HTTP_TIMEOUT
            ) as client:
                q = "veterinary" if type == "vet" else "market"
                params = {
                    "format": "jsonv2",
                    "q": q,
                    "bounded": 1,
                    "viewbox": f"{left},{top},{right},{bottom}",
                    "limit": limit,
                }
                r = await client.get("https://nominatim.openstreetmap.org/search", params=params)
                if r.status_code == 200:
                    js = r.json()
                    for i, it in enumerate(js[:limit]):
                        payload.append({
                            "id": 10_000_000 + i,
                            "name": it.get("display_name") or it.get("name"),
                            "lat": float(it.get("lat")),
                            "lon": float(it.get("lon")),
                            "category": type,
                            "tags": {"source": "nominatim", "class": str(it.get("class")), "type": str(it.get("type"))},
                        })
        except Exception as e:
            print(f"[places] nominatim error: {e}")

    # only cache non-empty results
    if payload:
        _PLACES_CACHE[key] = (now, payload)
    else:
        print(f"[places] not caching empty result for {type}")

    print(f"[places] type={type} r={radius_m} -> {len(payload)} results (limit={limit})")
    return payload

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

# ================= Gemini =================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
print(GEMINI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _gem = genai.GenerativeModel("gemini-2.5-flash")  # fast & economical
else:
    _gem = None

@app.post("/ai/crossbreed")
def ai_crossbreed(req: AiReq):
    if not _gem: return {"error":"GEMINI_API_KEY not set"}
    prompt = f"""
You are an expert cattle breeding assistant in India.
Breed: {req.breed}
Location: {req.location or 'unknown'}

List 3-5 compatible crossbreeds with {req.breed}. For each:
- compatibility reason
- pros/cons (health, yield, climate fit)
- cost/economic feasibility (rough idea)

Return concise bullet points.
"""
    out = _gem.generate_content(prompt)
    return {"text": out.text}

@app.post("/ai/market")
def ai_market(req: AiReq):
    if not _gem: return {"error":"GEMINI_API_KEY not set"}
    prompt = f"""
You are a livestock market advisor.
Breed: {req.breed}
Location: {req.location or 'unknown'}

Give:
- current indicative price range for {req.breed} (in INR) with assumptions
- 3-5 negotiation tips
- 3 nearby place types to sell (mandis/co-ops), not specific names.

Be concise. Mention uncertainty.
"""
    out = _gem.generate_content(prompt)
    return {"text": out.text}

@app.post("/ai/vaccinations")
def ai_vaccinations(req: AiReq):
    if not _gem: return {"error":"GEMINI_API_KEY not set"}
    prompt = f"""
You are a veterinary assistant.
Breed: {req.breed}
Region: {req.location or 'unknown'}

Provide:
- Core vaccinations (names, typical schedule windows)
- Parasite control and general care tips
- Any breed/regional caveats

Keep terse & practical. This is not medical advice.
"""
    out = _gem.generate_content(prompt)
    return {"text": out.text}

class DairyReq(BaseModel):
    breed: str | None = None
    location: str | None = None
    extra: str | None = None

@app.post("/ai/dairy_market")
def ai_dairy(req: DairyReq):
    if not _gem: 
        return {"error": "GEMINI_API_KEY not set"}
    prompt = f"""
You are a dairy procurement advisor for Indian farmers.
Location: {req.location or 'unknown'}

Give concise bullets:
- Nearby well-known dairies/brands (e.g., Amul, Mother Dairy, Country Delight, Nandini, Hatsun, Verka) that typically procure milk via collection centers
- The kind(s) of milk they commonly buy (cow/buffalo/mixed), common SNF/fat expectations (indicative)
- Typical delivery process (collection center timings, chilling, quality testing basics)
- Price notes are indicative and vary; mention uncertainty.

Keep it short. Avoid guarantees.
"""
    out = _gem.generate_content(prompt)
    return {"text": out.text}

# --- NEW: Translate to Hindi for TTS -----------------------------------------
@app.post("/ai/translate_hi")
def ai_translate_hi(req: AiReq):
    """
    Translate arbitrary text (sent in req.breed) into natural spoken Hindi,
    stripping markdown/URLs/parenthetical noise. Returns {"text": "..."}.
    """
    if not _gem:
        return {"error": "GEMINI_API_KEY not set"}

    # We reuse AiReq.breed as the input text (matches your ApiService.ai signature)
    src = (req.breed or "").strip()
    if not src:
        return {"error": "No text provided to translate"}

    prompt = f"""
You are a professional Hindi translator and copy editor.
Task: Translate the text into natural, conversational Hindi suitable for text-to-speech.
Rules:
- Remove markdown symbols, URLs, and any bracketed/parenthetical content.
- Keep bullets as short, simple lines if present.
- Output ONLY the final Hindi text. No preface, no notes, no quotes.

Text:
{src}
"""
    try:
        out = _gem.generate_content(prompt)
        text = (getattr(out, "text", "") or "").strip()
        return {"text": text}
    except Exception as e:
        return {"error": f"translate_hi failed: {e}"}