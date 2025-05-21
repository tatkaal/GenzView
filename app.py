#!/usr/bin/env python3
# ================================================================
#  GenzView Streamlit Studio – end-to-end demo (fixed version)
#  ---------------------------------------------------------------
#  Run with:
#
#       streamlit run appv2.py
#
#  – Fixes:
#      • asyncio “no running event loop”
#      • torch.classes watcher warnings
#      • uniform 140 × 140 thumbnails
#      • bounded main / heat-map / OCR image widths
#      • proper colour swatch (raw-HTML table)
#      • sidebar demographic rendered with pure Markdown
# ================================================================

# -----------------------------------------------------------------
#  Bootstrap: ensure an event loop exists (Streamlit/Windows quirk)
# -----------------------------------------------------------------
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

# -----------------------------------------------------------------
#  Suppress torch C++ spam & prevent Streamlit probing torch.classes
# -----------------------------------------------------------------
import os, warnings, logging
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings(
    "ignore",
    message=r"Tried to instantiate class .*torch\.classes.*"
)

import torch          # after suppressing warnings
torch.classes.__path__ = []          # silence “Examining the path…” logs
logging.getLogger("torch").setLevel(logging.ERROR)

# -----------------------------------------------------------------
#  Standard libs
# -----------------------------------------------------------------
from pathlib import Path
from typing   import Dict, Any, List, Tuple

# -----------------------------------------------------------------
#  Third-party
# -----------------------------------------------------------------
import numpy as np
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------
#  Your project imports
# -----------------------------------------------------------------
from genzview import GenzView, DemographicProfile     # core engine

# -----------------------------------------------------------------
#  Session-state helper
# -----------------------------------------------------------------
def ss_get(key, default=None):
    """Return st.session_state[key], initialising with default if missing."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# -----------------------------------------------------------------
#  Constant demographic profile
# -----------------------------------------------------------------
DEMO = DemographicProfile()

# -----------------------------------------------------------------
#  Page config & heading
# -----------------------------------------------------------------
st.set_page_config(
    page_title="GenzView Studio",
    page_icon="🎨",
    layout="wide",
)
st.title("🎨 GenzView Studio – View Packaging through Genz Eyes")

st.markdown(
"""
Interactively explore how **Urban Gen-Z trendsetters** interpret your cosmetic
packaging – and what design tweaks can nudge them to *add-to-cart*.

Select a thumbnail (or upload your own), then step through the analysis with
the ➡️ **Next** buttons. No code or guess-work – just a clear, scroll-friendly
story of what the AI is doing and *why*.
"""
)

# -----------------------------------------------------------------
#  0. Image chooser / uploader
# -----------------------------------------------------------------
ASSETS_DIR = Path("assets")
SAMPLES: List[Path] = sorted(
    p for p in ASSETS_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

cols = st.columns(3)
for idx, p in enumerate(SAMPLES):
    col = cols[idx % 3]
    with col:
        img = Image.open(p).resize((140, 140))
        if st.button("", key=f"thumb_{idx}", help=f"Select {p.name}"):
            st.session_state.selected_image = str(p)
        st.image(img, caption=p.name, width=140)

uploaded = st.file_uploader("…or upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    tmp_path = Path(f"tmp_upload.{uploaded.type.split('/')[-1]}")
    tmp_path.write_bytes(uploaded.read())
    st.session_state.selected_image = str(tmp_path)

sel_path = ss_get("selected_image", str(SAMPLES[0]))
st.image(sel_path, caption="Current selection", width=300)

# -----------------------------------------------------------------
#  1. Metadata inputs
# -----------------------------------------------------------------
st.subheader("📝 Packaging Metadata")
c1, c2, c3 = st.columns(3)
with c1:
    prod_cat = st.selectbox(
        "Product category",
        ["Facial Serum", "Cream", "Toner", "Cleanser", "Mask", "Lip Tint", "Other"],
        key="prod_cat",
    )
with c2:
    brand_tone = st.selectbox(
        "Brand tone",
        ["Premium", "Playful", "Eco", "Clinical", "Minimal"],
        key="brand_tone",
    )
with c3:
    brand_name = st.text_input("Brand name (optional)", key="brand_name")

META: Dict[str, Any] = {"product_category": prod_cat, "brand_tone": brand_tone}
if brand_name:
    META["brand_name"] = brand_name

# -----------------------------------------------------------------
#  Sidebar: demographic profile (pure Markdown)
# -----------------------------------------------------------------
with st.sidebar:
    st.header("🎯 Target Demographic")
    st.markdown(f"""
- **Age:** {DEMO.age_range[0]}–{DEMO.age_range[1]}
- **Gender:** {DEMO.gender}
- **Region:** {DEMO.region}
- **Traits:** {', '.join(DEMO.traits)}
- **Values:** {', '.join(DEMO.values)}
- **Digital Habits:** TikTok · Instagram · beauty influencers
""")

# -----------------------------------------------------------------
#  Lazy initialisation of GenzView engine
# -----------------------------------------------------------------
def get_tl() -> GenzView:
    if "tl" not in st.session_state:
        st.session_state.tl = GenzView(demo=DEMO)
    return st.session_state.tl

# -----------------------------------------------------------------
#  STEP 0 → 1 : Start analysis
# -----------------------------------------------------------------
if st.button("🚀 Start Analysis", disabled=ss_get("step", 0) >= 1):
    tl       = get_tl()
    feats    = tl._analyse(sel_path, META)
    ocr_path, words = tl._ocr_overlay(sel_path) if tl._ocr else (None, [])
    heat_path = tl._heatmap(sel_path, feats, words, META)

    st.session_state.update({
        "step":      1,
        "feats":     feats,
        "words":     words,
        "heat_path": heat_path,
        "ocr_path":  ocr_path,
    })
    st.rerun()

# -----------------------------------------------------------------
#  STEP 1 : attention heat-map
# -----------------------------------------------------------------
if ss_get("step") >= 1:
    st.header("👀 Where does Gen-Z attention land first?")
    st.image(
        ss_get("heat_path"),
        caption="Red–yellow = high influence on perception",
        width=600,
    )
    if st.button("Next ➡️ Explain extracted features", key="go_feat",
                 disabled=ss_get("step") >= 2):
        st.session_state.step = 2
        st.rerun()

# -----------------------------------------------------------------
#  STEP 2 : feature extraction
# -----------------------------------------------------------------
if ss_get("step") >= 2:
    st.header("🧬 Visual features extracted")
    feats = ss_get("feats")

    dom_bgr = feats["dominant_color"].astype(int)
    dom_hex = "#%02x%02x%02x" % tuple(int(c) for c in dom_bgr[::-1])
    pastel  = feats["pastel_palette"]
    has_glass = feats["has_glass"]
    sym   = feats["symmetry"]
    ted   = feats["text_edge_density"]
    tar   = feats["text_area_ratio"] * 100

    # raw-HTML table so colour swatch renders
    st.markdown(f"""
<table style="width:100%;border-collapse:collapse">
  <thead>
    <tr><th style="text-align:left">Feature</th>
        <th style="text-align:left">Value</th>
        <th style="text-align:left">Why it matters</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Dominant colour</b></td>
      <td><span style="background:{dom_hex};border:1px solid #666;
          display:inline-block;width:30px;height:15px"></span>&nbsp;{dom_hex}</td>
      <td>Soft pastels score <b>+2 Aesthetic</b></td>
    </tr>
    <tr><td><b>Pastel palette</b></td><td>{pastel}</td>
        <td>Pastel look = IG-ready</td></tr>
    <tr><td><b>Detected glass</b></td><td>{has_glass}</td>
        <td>Glass boosts <b>+3 Luxury</b></td></tr>
    <tr><td><b>Symmetry</b></td><td>{sym:.2f}</td>
        <td>&gt;0.8 looks premium</td></tr>
    <tr><td><b>Edge density</b></td><td>{ted:.3f}</td>
        <td>Dense edges hurt clarity</td></tr>
    <tr><td><b>Text area&nbsp;%</b></td><td>{tar:.2f}%</td>
        <td>Too much text feels noisy</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

    if ss_get("ocr_path"):
        st.image(ss_get("ocr_path"), caption="OCR overlay", width=600)
    else:
        st.info("OCR unavailable – skipping text overlay.")

    if st.button("Next ➡️ Calculate heuristic score", key="go_score",
                 disabled=ss_get("step") >= 3):

        # --- heuristic scoring ------------------------------------------------
        s = {"aesthetic": 5, "trust": 5, "luxury": 5}
        contribs: List[Tuple[str, int, str]] = []

        def add(rule: str, amt: int, targets: List[str]):
            for t in targets:
                s[t] += amt
            contribs.append((rule, amt, ", ".join(targets)))

        # colour / material / geometry rules
        if pastel:            add("Pastel palette", +2, ["aesthetic"])
        if has_glass:         add("Glass bottle",   +3, ["luxury"])
        if sym > .8:          add("High symmetry",  +1, ["aesthetic"])
        if ted > .15:         add("Busy edges",     -1, ["aesthetic", "trust"])
        if tar > 3:           add("Lots of text",   -1, ["aesthetic", "trust"])
        if META["brand_tone"].lower() == "premium":
                              add("Premium tone",   +1, ["luxury"])

        # OCR keywords
        words = ss_get("words")
        good_kw = ["vitamin", "hyaluronic", "organic", "glow", "clean", "spf"]
        bad_kw  = ["paraben", "sulfate", "harsh"]
        add_good = sum(w in words for w in good_kw)
        add_bad  = sum(w in words for w in bad_kw)
        if add_good: add("Positive keywords", +add_good, ["trust"])
        if add_bad:  add("Negative keywords", -add_bad,  ["trust"])

        if ("instagrammable" in (v.lower() for v in DEMO.values)) and pastel:
            add("Pastel aligns w/ value", +1, ["aesthetic"])
        if META["brand_tone"].lower() == "minimal" and tar < 2:
            add("Minimal text aligns", +1, ["aesthetic"])

        for k in s:
            s[k] = max(0, min(10, s[k]))
        purchase = round((s["aesthetic"] + s["trust"] + s["luxury"]) / 3, 2)

        scores = {**s, "purchase": purchase}
        st.session_state.update({
            "scores":   scores,
            "contribs": contribs,
            "step":     3,
        })
        st.rerun()

# -----------------------------------------------------------------
#  STEP 3 : heuristic score + rule breakdown
# -----------------------------------------------------------------
if ss_get("step") >= 3:
    st.header("📊 Heuristic perception score")
    sc = ss_get("scores")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Aesthetic", f"{sc['aesthetic']:.1f}")
    m2.metric("Trust",     f"{sc['trust']:.1f}")
    m3.metric("Luxury",    f"{sc['luxury']:.1f}")
    m4.metric("Purchase",  f"{sc['purchase']:.1f}")

    with st.expander("How each rule moved the needle"):
        md = "| Rule | Impact | Attribute(s) |\n|------|:------:|--------------|\n"
        for r, amt, tgt in ss_get("contribs"):
            sign = "➕" if amt > 0 else "➖"
            md += f"| {r} | {sign}{abs(amt)} | {tgt} |\n"
        st.markdown(md)

    if st.button("Get Recommendations 💡", disabled=ss_get("step") >= 4):
        rec = get_tl()._recommend(
            type("S", (), sc)(), ss_get("feats"), META, ss_get("words")
        )
        st.session_state.update({"recommendation": rec, "step": 4})
        st.rerun()

# -----------------------------------------------------------------
#  STEP 4 : heuristic recommendations
# -----------------------------------------------------------------
if ss_get("step") >= 4:
    st.header("💡 Design recommendations (heuristic)")
    st.success(ss_get("recommendation"))

    if st.button("Try ML scoring 🚀", disabled=ss_get("step") >= 5):
        tl        = get_tl()
        ml_delta  = tl._ml_delta(sel_path)
        sc_ml     = sc.copy()
        for k in ["aesthetic", "trust", "luxury", "purchase"]:
            sc_ml[k] = max(0, min(10, sc_ml[k] + ml_delta[k]))

        tl.last_scores = type("S", (), sc_ml)()
        ml_rec = tl._recommend(tl.last_scores, ss_get("feats"), META, ss_get("words"))

        st.session_state.update({
            "ml_scores": sc_ml,
            "ml_delta":  ml_delta,
            "ml_rec":    ml_rec,
            "step":      5,
        })
        st.rerun()

# -----------------------------------------------------------------
#  STEP 5 : ML-augmented score + recommendations
# -----------------------------------------------------------------
if ss_get("step") >= 5:
    st.header("🤖 ML-augmented perception score")
    sc2 = ss_get("ml_scores")
    dl  = ss_get("ml_delta")
    m1, m2, m3, m4 = st.columns(4)
    for col, label, key in zip(
        (m1, m2, m3, m4),
        ("Aesthetic", "Trust", "Luxury", "Purchase"),
        ("aesthetic", "trust", "luxury", "purchase")
    ):
        col.metric(label, f"{sc2[key]:.1f}", delta=f"{dl[key]:+0.2f}")

    st.success(ss_get("ml_rec"))

    if st.button("Generate Variants 🎨", disabled=ss_get("step") >= 6):
        variants = get_tl().generate_variants(sel_path, num=4, source="openai")
        st.session_state.update({"variants": variants, "step": 6})
        st.rerun()

# -----------------------------------------------------------------
#  STEP 6 : AI-generated variants
# -----------------------------------------------------------------
if ss_get("step") >= 6:
    st.header("🎨 AI-generated variants (Stable Diffusion)")
    variants = ss_get("variants")
    vcols = st.columns(len(variants))
    for col, vpath in zip(vcols, variants):
        col.image(vpath, width=300)
        with open(vpath, "rb") as f:
            col.download_button("Download", f.read(), file_name=Path(vpath).name)

# -----------------------------------------------------------------
#  Footer
# -----------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.caption("© 2025 • GenzView Studio – technical demo (no code-peek needed)")
