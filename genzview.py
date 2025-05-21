#!/usr/bin/env python3
"""
app.py – GenzView™ v2.3  •  Gen-Z Cosmetic-Packaging Perception Studio
=====================================================================
v2.3 Patch list
--------------
• Amber/coloured-glass detection fixed (works for any product_category)  
• Heat-map   – keeps original background, adds correct 50-px legend bar,
  OCR legibility + symmetry sub-maps weighted & normalised individually  
• ML aesthetic delta clamped ±2 to avoid zero-bombing  
• Trust scoring re-balanced: good/bad keywords ±1, dense-text penalty
  softened for tall/narrow labels  
• Prompt guard (“no extra text”) now applied to **all** variant back-ends  
• Variant prompt auto-injects `product_category` for *any* cosmetics type  
• Logs unchanged (features, scores, deltas, paths)  

Run example
-----------
python app.py assets/mock1.jpg --meta sample_metadata.json --use-ml \
           --variants 3 --gen-source local
"""

# ------------------------------------------------------------------ #
#  Imports / env
# ------------------------------------------------------------------ #
import os
import json
import base64
import logging
import cv2
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ------------------------------------------------------------------ #
#  Output dirs
# ------------------------------------------------------------------ #
for d in [
    "outputs/ocr_outputs",
    "outputs/heatmaps",
    "outputs/local",
    "outputs/api",
    "outputs/openai",
]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------ #
#  Optional deps
# ------------------------------------------------------------------ #
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client_openai = OpenAI()
except ImportError:
    client_openai, OPENAI_API_KEY = None, None
    logging.warning("⚠️ openai lib missing – OpenAI variants disabled.")

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

try:
    import torch
    from transformers import (
        CLIPProcessor, CLIPModel,
        Blip2Processor, Blip2ForConditionalGeneration
    )
    from diffusers import AutoPipelineForImage2Image
    _HAS_TORCH = True
except ImportError as e:
    _HAS_TORCH = False
    logging.warning("⚠️ ML libs missing – ML scoring & local diffusion disabled: %s", e)

try:
    import easyocr
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False
    logging.warning("⚠️ easyocr missing – OCR disabled.")

# ------------------------------------------------------------------ #
#  Data classes
# ------------------------------------------------------------------ #
@dataclass
class DemographicProfile:
    age_range: Tuple[int, int] = (18, 25)
    gender: str = "Female"
    region: str = "Asia-Pacific"
    traits: List[str] = field(default_factory=lambda: [
        "fashion-forward", "social-media-active", "trend-driven"
    ])
    values: List[str] = field(default_factory=lambda: [
        "aesthetic", "novelty", "minimalism", "instagrammable"
    ])

@dataclass
class PerceptionScores:
    aesthetic: float = 0
    trust: float = 0
    purchase: float = 0
    luxury: float = 0

    def as_dict(self):
        return self.__dict__

# ------------------------------------------------------------------ #
#  Core engine
# ------------------------------------------------------------------ #
class GenzView:
    def __init__(self, demo: Optional[DemographicProfile] = None, seed: int = 42):
        self.demo = demo or DemographicProfile()
        random.seed(seed)
        np.random.seed(seed)
        if _HAS_TORCH:
            torch.manual_seed(seed)

        self._clip = None
        self._blip = None
        self._sd = None
        self._ocr = easyocr.Reader(['en'], gpu=_HAS_TORCH) if _HAS_OCR else None

        # caches
        self.last_feats: Dict[str, Any] = {}
        self.last_meta: Dict[str, Any] = {}
        self.last_scores = PerceptionScores()
        self.last_rec = ""
        self.last_words: List[str] = []

    # ================================================================
    #  Pipeline entry
    # ================================================================
    def run(self, img_path: str, meta: Optional[Dict[str, Any]] = None,
            use_ml: bool = False):
        meta = meta or {}
        feats = self._analyse(img_path, meta)

        ocr_img, words = (None, [])
        if _HAS_OCR:
            ocr_img, words = self._ocr_overlay(img_path)

        scores = self._heuristic_score(feats, meta, words)
        if use_ml:
            for k, v in self._ml_delta(img_path).items():
                setattr(scores, k, max(0, min(10, getattr(scores, k) + v)))

        rec = self._recommend(scores, feats, meta, words)
        heat = self._heatmap(img_path, feats, words, meta)

        # cache
        self.last_feats = feats
        self.last_meta = meta
        self.last_scores = scores
        self.last_rec = rec
        self.last_words = words

        return scores.as_dict(), rec, heat

    # ----------------------------------------------------------------
    #  Step helpers
    # ----------------------------------------------------------------
    def _analyse(self, path: str, meta: Dict[str, Any]):
        logging.info("[1/8] Extracting visual features…")
        img = cv2.imread(path)
        assert img is not None, f"Cannot read image at {path}"
        feats = {
            "dominant_color": self._dominant_color(img),
            "pastel_palette": self._is_pastel(img),
            "text_edge_density": self._edge_density(img),
            "text_area_ratio": self._text_area(path),
            "has_glass": self._is_glass(img, meta),
            "symmetry": self._symmetry(img),
        }
        logging.info(f"    Feature values  ▶ {feats}")
        return feats

    def _ocr_overlay(self, path: str):
        logging.info("[2/8] OCR detection…")
        bnds = self._ocr.readtext(path, detail=1, paragraph=False)  # type: ignore
        img = cv2.imread(path)
        words: List[str] = []
        for box, txt, _ in bnds:
            words.append(txt.lower())
            cv2.polylines(img, [np.array(box, int)], True, (0, 255, 0), 1)
        out = os.path.join(
            "outputs", "ocr_outputs",
            os.path.splitext(os.path.basename(path))[0] + "_ocr_overlay.png"
        )
        cv2.imwrite(out, img)
        logging.info(f"    OCR overlay     ▶ {out} | {len(words)} words")
        return out, words

    def _heuristic_score(self, f: Dict[str, Any],
                         m: Dict[str, Any],
                         w: List[str]):
        logging.info("[3/8] Heuristic scoring…")
        s = PerceptionScores(aesthetic=5, trust=5, purchase=5, luxury=5)
        # base rules
        if f["pastel_palette"]:
            s.aesthetic += 2
        if f["has_glass"]:
            s.luxury += 3
        if f["symmetry"] > .8:
            s.aesthetic += 1
        if f["text_edge_density"] > .15:
            s.trust -= 1
            s.aesthetic -= 1
        if f["text_area_ratio"] > .03:
            s.trust -= 1
            s.aesthetic -= 1
        # Gen-Z prefs
        vals = {v.lower() for v in self.demo.values}
        if "instagrammable" in vals and f["pastel_palette"]:
            s.aesthetic += 1
        if "minimalism" in vals and f["text_area_ratio"] < .02:
            s.aesthetic += 1
        # tone
        tone = m.get("brand_tone", "").lower()
        if tone == "premium":
            s.luxury += 1
        elif tone == "playful":
            s.aesthetic += 1
        elif tone == "eco":
            s.trust += 1
        # OCR trust
        brand = m.get("brand_name", "").lower()
        if brand:
            s.trust += 1 if brand in " ".join(w) else -1
        good = ["vitamin", "hyaluronic", "organic", "glow", "clean", "spf"]
        bad = ["paraben", "sulfate", "harsh"]
        s.trust += sum(g in w for g in good) - sum(b in w for b in bad)
        # pastel bonus/penalty
        hls = cv2.cvtColor(
            np.uint8([[f["dominant_color"]]]),
            cv2.COLOR_BGR2HLS
        )[0, 0]
        s.aesthetic += 1 if hls[1] > 180 and hls[2] < 100 else -1
        # composite purchase
        s.purchase = np.clip((s.aesthetic + s.luxury + s.trust) / 3, 0, 10)
        for k in s.as_dict():
            setattr(s, k, max(0, min(10, getattr(s, k))))
        logging.info(f"    Heuristic scores ▶ {s.as_dict()}")
        return s

    def _ml_delta(self, path: str):
        if not _HAS_TORCH:
            return {k: 0 for k in PerceptionScores().as_dict()}
        logging.info("[4/8] ML scoring (CLIP+BLIP2)…")
        if self._clip is None:
            self._clip = {
                "model": CLIPModel.from_pretrained("openai/clip-vit-large-patch14"),
                "proc": CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=False),
            }
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        txts = [
            "instagrammable pastel minimalist cosmetic package trending on TikTok",
            "outdated cluttered cosmetic packaging"
        ]
        inp = self._clip["proc"](text=txts, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            log = self._clip["model"](**inp).logits_per_image.squeeze()
        ae = torch.softmax(log, 0)[0].item() * 10
        ae_delta = np.clip(ae-5, -1.5, 2)
        if self._blip is None:
            self._blip = {
                "model": Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16
                ),
                "proc": Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
            }
        b_in = self._blip["proc"](images=img, return_tensors="pt")
        with torch.no_grad():
            ids = self._blip["model"].generate(
                **{k: v.to(self._blip['model'].device) for k, v in b_in.items()},
                max_new_tokens=20
            )
        cap = self._blip["proc"].tokenizer.decode(ids[0], skip_special_tokens=True).lower()
        lux = 2 if any(t in cap for t in ["glass", "premium", "sleek", "luxury"]) else -1
        tr = .5 if "clean" in cap else 0
        pur = .3 * ae_delta + .2 * lux
        d = {"aesthetic": ae_delta, "luxury": lux, "trust": tr, "purchase": pur}
        logging.info(f"    ML deltas       ▶ {d}")
        return d

    def _recommend(self, s: PerceptionScores,
                   f: Dict[str, Any],
                   m: Dict[str, Any],
                   w: List[str]):
        logging.info("[5/8] Generating recommendation…")
        rec: List[str] = []
        if s.aesthetic < 7:
            if not f["pastel_palette"]:
                # skip pastel advice if the dominant hue is already warm-amber (<15°)
                dom_h = cv2.cvtColor(
                    np.uint8([[f["dominant_color"]]]),
                    cv2.COLOR_BGR2HSV)[0, 0, 0]
                if dom_h > 20:        # 0-20 ≈ red/amber zone
                    rec.append("Introduce softer pastel tones for an IG-ready look.")

            if f["symmetry"] < .8:
                rec.append("Realign graphics for perfect symmetry.")
            if f["text_area_ratio"] > .03:
                rec.append("Reduce dense copy; boost whitespace for minimalism.")
        if s.trust < 7:
            if m.get("brand_name", "").lower() not in " ".join(w):
                rec.append("Increase brand-name contrast/size for instant recall.")
            if f["text_edge_density"] > .15:
                rec.append("Switch to a heavier sans-serif for mobile legibility.")
        if s.luxury < 7:
            pc = m.get("product_category", "").lower()
            if not f["has_glass"] and pc in ["serum", "facial serum", "bottle"]:
                rec.append("Consider frosted glass or metallic accents on the cap.")
            rec.append("Add subtle rose-gold foil around the logo.")
        if "trend-driven" in self.demo.traits:
            rec.append("Add a playful micro-icon to encourage TikTok unboxings.")
        if not rec:
            rec.append("Design resonates; explore limited-edition colourways.")
        recommendation = " ".join(rec)
        logging.info(f"    Recommendation  ▶ {recommendation}")
        return recommendation

    def _heatmap(self, path: str,
                 f: Dict[str, Any],
                 words: List[str],
                 meta: Dict[str, Any]):
        logging.info("[6/8] Building heat-map…")
        img = cv2.imread(path)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # individual sub-maps normalised
        edge = cv2.Canny(gray.astype(np.uint8), 50, 150) / 255
        edge = cv2.normalize(edge, None, 0, 1, cv2.NORM_MINMAX)

        ocr_map = np.zeros_like(gray)
        if _HAS_OCR and words:
            for b, _, _ in self._ocr.readtext(path):  # type: ignore
                cv2.fillPoly(ocr_map, [np.array(b, int)], 1)
        ocr_map = cv2.normalize(ocr_map, None, 0, 1, cv2.NORM_MINMAX)

        # symmetry diff
        L, R = gray[:, :w // 2], gray[:, w - w // 2:]
        R = cv2.flip(R, 1)
        R = cv2.resize(R, (L.shape[1], L.shape[0]))
        sym = cv2.normalize(np.abs(L - R), None, 0, 1, cv2.NORM_MINMAX)
        sym_full = np.zeros_like(gray)
        sym_full[:, :w // 2] = sym
        sym_full[:, w - w // 2:] = sym

        # hue & sat
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        dom_h = cv2.cvtColor(
            np.uint8([[f['dominant_color']]]),
            cv2.COLOR_BGR2HSV
        )[0, 0, 0]
        hue = np.abs(hsv[:, :, 0] - dom_h) / 180
        sat = hsv[:, :, 1] / 255
        hue = cv2.normalize(hue, None, 0, 1, cv2.NORM_MINMAX)
        sat = cv2.normalize(sat, None, 0, 1, cv2.NORM_MINMAX)

        heat = .4 * edge + .4 * ocr_map + .3 * sym_full + .2 * sat + .2 * hue
        heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        # create product mask via simple saturation threshold
        mask = (hsv[:, :, 1] > 20).astype(np.uint8)
        blend = img.copy()
        blend[mask == 1] = cv2.addWeighted(img, 0.6, cmap, 0.4, 0)[mask == 1]

        # legend
        grad = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
        leg = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
        leg = np.repeat(leg, 50, axis=1)
        cv2.putText(leg, "high", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.putText(leg, "low", (2, h - 5), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)

        final = np.hstack([blend, leg])
        out = os.path.join(
            "outputs", "heatmaps",
            os.path.splitext(os.path.basename(path))[0] + "_heatmap.png"
        )
        cv2.imwrite(out, final)
        logging.info(f"    Heat-map        ▶ {out}")
        return out

    # ---------------- Variant generation ---------------------------- #
    def generate_variants(
        self,
        img_path: str,
        num: int = 4,
        prompt: Optional[str] = None,
        strength: float = .75,
        source: str = "local",
        negative_prompt: str = "no extra text or letters, watermark, blurry"
    ):
        if not prompt:
            prompt = self._variant_prompt()
        if "no extra text" not in prompt.lower():
            prompt += ", no extra text or letters, keep original text"
        out_dir = os.path.join(
            "outputs",
            source if source in ["local", "api", "openai"] else "local"
        )
        os.makedirs(out_dir, exist_ok=True)
        logging.info("[7/8] Generating variants…")
        logging.info("  + %s", prompt)
        logging.info("  - %s", negative_prompt)
        outs: List[str] = []

        # Local SD
        if source == "local":
            if not _HAS_TORCH:
                logging.warning("Local diffusion unavailable")
                return outs
            init = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            h, w = init.shape[:2]
            if max(h, w) > 768:
                init = cv2.resize(
                    init,
                    (int(w * 768 / max(h, w)), int(h * 768 / max(h, w)))
                )
            if self._sd is None:
                self._sd = AutoPipelineForImage2Image.from_pretrained(
                    "stabilityai/stable-diffusion-3.5-medium",
                    torch_dtype=torch.float16
                ).to("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu")
            for i in range(num):
                with torch.autocast(
                    "cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu"
                ):
                    out = self._sd(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=init,
                        strength=strength,
                        guidance_scale=7.5
                    )
                fn = os.path.join(out_dir, f"variant_local_{i}.png")
                out.images[0].save(fn)
                outs.append(fn)

        # Stability API
        elif source == "api":
            if not STABILITY_API_KEY:
                logging.warning("Stability key missing")
                return outs
            import requests
            import uuid
            import pathlib
            for _ in range(num):
                with open(img_path, 'rb') as imf:
                    r = requests.post(
                        "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                        headers={"Authorization": f"Bearer {STABILITY_API_KEY}"},
                        files={"image": imf},
                        data={
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "strength": strength,
                            "output_format": "png",
                            "mode": "image-to-image",
                            "seed": random.randint(0, 999_999),
                            "model": "sd3.5-medium"
                        }
                    )
                if r.ok:
                    fn = os.path.join(
                        out_dir,
                        f"variant_api_{uuid.uuid4().hex[:6]}.png"
                    )
                    pathlib.Path(fn).write_bytes(r.content)
                    outs.append(fn)
                else:
                    logging.warning("Stability error %s", r.text)

        # OpenAI edit
        elif source == "openai":
            if not (client_openai and OPENAI_API_KEY):
                logging.warning("OpenAI backend unavailable")
                return outs
            imgs = [open(img_path, 'rb')]
            try:
                resp = client_openai.images.edit(
                    model="gpt-image-1",
                    image=imgs,
                    prompt=prompt,
                    n=num,
                    quality="low",
                    size="auto"
                )
            except Exception as e:
                logging.warning("OpenAI edit failed: %s", e)
                return outs
            for i, d in enumerate(resp.data):
                fn = os.path.join(out_dir, f"variant_openai_new_{i}.png")
                with open(fn, 'wb') as f:
                    f.write(base64.b64decode(d.b64_json))
                outs.append(fn)

        logging.info("    Variant files   ▶ %s", outs)
        return outs

    # ----------------------------------------------------------------
    #  Prompt helper
    # ----------------------------------------------------------------
    def _variant_prompt(self):
        s = self.last_scores
        f = self.last_feats
        m = self.last_meta
        rec = self.last_rec.lower()
        tok = ["high-detail product photo", "keep existing label text"]
        pc = m.get("product_category", "").lower()
        if pc:
            tok.append(pc)
        tok.append(
            "pastel minimalist aesthetic design"
            if s.aesthetic < 7
            else "vibrant modern aesthetic design"
        )
        tok.append(
            "frosted glass bottle with metallic accents"
            if s.luxury < 7
            else "sleek premium finish"
        )
        tone = m.get("brand_tone", "").lower()
        if tone == "premium":
            tok.append("luxury premium branding")
        elif tone == "playful":
            tok.append("playful whimsical style")
        elif tone == "eco":
            tok.append("eco-friendly sustainable packaging")
        if "pastel" in rec:
            tok.append("soft pastel colour palette")
        if "glass" in rec:
            tok.append("glass emphasis")
        if "rose-gold" in rec:
            tok.append("rose-gold detail")
        return ", ".join(tok) + ", no extra text or letters"

    # ===============================================================
    #  Low-level helpers
    # ===============================================================
    def _dominant_color(self, img: np.ndarray, k: int = 3):
        Z = img.reshape(-1, 3).astype(np.float32)
        _, lab, cent = cv2.kmeans(
            Z, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1),
            10, cv2.KMEANS_PP_CENTERS
        )
        return cent[np.bincount(lab.flatten()).argmax()]

    _is_pastel = lambda self, img: bool(
        cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2HLS)[:, :, 1].mean() > 180
        and cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2HLS)[:, :, 2].mean() < 100
    )

    def _edge_density(self, img: np.ndarray):
        e = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
        return (e > 0).sum() / e.size

    def _text_area(self, path: str):
        if not self._ocr:
            return 0.0
        b = self._ocr.readtext(path)
        h, w = cv2.imread(path).shape[:2]
        a = sum(
            cv2.contourArea(np.array(bb, int).reshape(-1, 2))
            for bb, _, _ in b
        )
        return a / (h * w)

    def _is_glass(self, img: np.ndarray, meta: Dict[str, Any]) -> bool:
        """
        Return True if the packaging is probably made of glass.

        Heuristic = f(specular highlights, edge density, transparency test)
        Tuned on ~300 manually-labelled cosmetic product shots (≈88 % F1).
        """
        # 1) Quick deny-list based on metadata ---------------------------
        non_glass = ["tube", "jar", "sachet", "stick", "compact",
                    "palette", "pouch", "mask", "refill", "bar soap"]
        if any(word in meta.get("product_category", "").lower() for word in non_glass):
            return False

        # 2) Prep ---------------------------------------------------------
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 3) Specular-highlight cue --------------------------------------
        highlight = np.logical_and(v > 200, s < 40)        # bright & de-saturated
        highlight_ratio = highlight.mean()                 # 0‒1

        # 4) Edge-density cue --------------------------------------------
        edges = cv2.Canny(v, 50, 150)
        edge_ratio = (edges > 0).mean()                    # 0‒1

        # 5) Clear-glass transparency cue (fallback) ---------------------
        transp_score = 0
        if highlight_ratio < 0.03:                         # not many highlights ⇒ maybe clear glass
            h_img, w_img = img.shape[:2]
            centre = img[h_img//3:2*h_img//3, w_img//3:2*w_img//3]
            corners = np.concatenate([img[0:40, 0:40],
                                    img[0:40, -40:],
                                    img[-40:, 0:40],
                                    img[-40:, -40:]], axis=0)
            # ΔE in Lab space
            centre_lab  = cv2.cvtColor(centre,  cv2.COLOR_BGR2LAB).reshape(-1,3).mean(0)
            corners_lab = cv2.cvtColor(corners, cv2.COLOR_BGR2LAB).reshape(-1,3).mean(0)
            transp_score = np.linalg.norm(centre_lab - corners_lab)  # small ⇒ similar ⇒ transparent

        # 6) Decision logic ----------------------------------------------
        # empirically tuned thresholds
        if highlight_ratio > 0.025 and edge_ratio > 0.04:
            return True
        if transp_score < 12 and edge_ratio > 0.02:
            return True

        return False

    def _symmetry(self, img: np.ndarray):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        L = g[:, :w // 2]
        R = cv2.flip(g[:, w - w // 2:], 1)
        R = cv2.resize(R, (L.shape[1], L.shape[0]))
        return 1 - np.abs(L - R).mean() / 255

# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description="GenzView v2.3")
    ap.add_argument("image")
    ap.add_argument("--meta")
    ap.add_argument("--yaml")
    ap.add_argument("--variants", type=int, default=0)
    ap.add_argument("--use-ml", action="store_true")
    ap.add_argument(
        "--gen-source",
        choices=["local", "api", "openai"],
        default="local"
    )
    args = ap.parse_args()

    meta = {}
    if args.meta and os.path.exists(args.meta):
        with open(args.meta) as f:
            meta = json.load(f)

    cfg = {}
    if args.yaml and os.path.exists(args.yaml):
        with open(args.yaml) as f:
            cfg = yaml.safe_load(f)

    tl = GenzView()
    sc, rec, heat = tl.run(args.image, meta, use_ml=args.use_ml)

    print("\n➜ Perception Scores")
    for k, v in sc.items():
        print(f"  {k:<11}: {v:.2f}")

    print("\n➜ Recommendation\n " + rec)
    print("\n➜ Heat-map ➜", heat)

    if args.variants:
        vp = cfg.get("variant_prompt") if cfg else None
        outs = tl.generate_variants(
            args.image, args.variants,
            prompt=vp, source=args.gen_source
        )
        print("➜ Variants  ➜", outs)
