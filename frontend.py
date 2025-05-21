#!/usr/bin/env python3
# ================================================================
#  🎨  GenzView Streamlit Studio
# ================================================================

import os
import warnings
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import streamlit as st
from PIL import Image
import torch

# -------- helpers --------
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def ss_get(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# -------- suppress torch chatter --------
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings(
    "ignore", message=r"Tried to instantiate class .*torch\.classes.*"
)
torch.classes.__path__ = []
logging.getLogger("torch").setLevel(logging.ERROR)

# -------- GenzView engine imports --------
from genzview import GenzView, DemographicProfile

# ----------------------------------------------------------------
#  Page config: centered layout
# ----------------------------------------------------------------
st.set_page_config(
    page_title="GenzView Studio",
    page_icon="🎨",
    layout="centered",
)
st.markdown(
    """
    <style>
      .block-container {
        max-width: 1300px;
        padding-left: 3rem;
        padding-right: 3rem;
      }

      /* Target most text containers and widgets */
      html, body, [class*="css"], .stMarkdown, .stText, .stHeader {
        font-size: 18px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------
#  Title & Intro
# ----------------------------------------------------------------
st.title("🎨 GenzView Studio – View Packaging through Genz Eyes")
st.markdown(
    "Interactively explore how **Urban Gen-Z trend-setters** interpret your "
    "cosmetic packaging – and what tweaks can *nudge them to add-to-cart*."
)

# ----------------------------------------------------------------
#  Demographic Card
# ----------------------------------------------------------------
DEMO = DemographicProfile()
with st.expander("🎯 Target Demographic", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
- **Age:** {DEMO.age_range[0]} – {DEMO.age_range[1]}
- **Gender:** {DEMO.gender}
- **Region:** {DEMO.region}
""")
    with c2:
        st.markdown(f"""
- **Traits:** {', '.join(DEMO.traits)}
- **Values:** {', '.join(DEMO.values)}
- **Digital Habits:** TikTok · Instagram · beauty influencers
""")

import base64
from pathlib import Path

def to_base64(img_path):
    data = Path(img_path).read_bytes()
    return base64.b64encode(data).decode()

# ----------------------------------------------------------------
#  Image Selection
# ----------------------------------------------------------------
ASSETS_DIR = Path("assets")
SAMPLES: List[Path] = sorted(
    p for p in ASSETS_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

with st.container(border=True):
    st.subheader("🖼️ Select Packaging Image")
    uploaded = st.file_uploader("…or upload an image", type=["png","jpg","jpeg"])
    if uploaded:
        tmp_path = Path(f"tmp_upload.{uploaded.type.split('/')[-1]}")
        tmp_path.write_bytes(uploaded.read())
        st.session_state.selected_image = str(tmp_path)
    else:
        default = ss_get("selected_file", SAMPLES[0].name)
        sel_name = st.selectbox(
            "Choose an image:",
            [p.name for p in SAMPLES],
            index=[p.name for p in SAMPLES].index(default),
            key="selected_file"
        )
        st.session_state.selected_image = str(ASSETS_DIR / sel_name)

    # Center and resize selected image
    st.markdown("#### Current selection", unsafe_allow_html=True)
    cols = st.columns([1,6,1])
    with cols[1]:
        st.markdown(
            f"<div style='text-align:center;'><img src='data:image/png;base64,{to_base64(st.session_state.selected_image)}' width='200'></div>",
            unsafe_allow_html=True
        )

# ----------------------------------------------------------------
#  Metadata Inputs
# ----------------------------------------------------------------
with st.container(border=True):
    st.subheader("📝 Packaging Metadata")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        prod_cat = st.selectbox(
            "Product category",
            ["Facial Serum","Cream","Body Lotion", "Toner","Cleanser","Mask","Lip Tint","Other"]
        )
    with m2:
        brand_tone = st.selectbox(
            "Brand tone",
            ["Premium","Playful","Eco","Clinical","Minimal"]
        )
    with m3:
        brand_name = st.text_input("Brand name (optional)")
    with m4:
        price_range = st.slider(
            "Price range ($)",
            0, 200, (20, 50), step=5
        )

    META: Dict[str,Any] = {
        "product_category": prod_cat,
        "brand_tone": brand_tone,
        "price_range": price_range
    }
    if brand_name:
        META["brand_name"] = brand_name

    # Show metadata usage
    st.markdown(
        f"**Metadata in use:**\n\n"
        f"- Category: **{META['product_category']}**  \n"
        f"- Tone: **{META['brand_tone']}**  \n"
        f"- Brand: **{META.get('brand_name','—')}**  \n"
        f"- Price Range: **${META['price_range'][0]}–${META['price_range'][1]}**"
    )

# ----------------------------------------------------------------
#  Engine & State Helpers
# ----------------------------------------------------------------
ss_get("step", 0)

def get_tl():
    if "tl" not in st.session_state:
        st.session_state.tl = GenzView(demo=DEMO)
    return st.session_state.tl

# ----------------------------------------------------------------
#  Auto-scroll injector
# ----------------------------------------------------------------
import streamlit.components.v1 as components
def do_scroll():
    tgt = st.session_state.get("scroll_target")
    if tgt:
        components.html(f"""
<script>
  parent.document.getElementById("{tgt}")
    ?.scrollIntoView({{behavior:"smooth",block:"start"}});
</script>""", height=0)
        del st.session_state["scroll_target"]

# ----------------------------------------------------------------
#  STEP 0 → 1: Start analysis
# ----------------------------------------------------------------
if st.button("🚀 Start analysis", disabled=st.session_state.step >= 1):
    tl = get_tl()
    sel = st.session_state.selected_image
    feats = tl._analyse(sel, META)
    ocr_path, words = tl._ocr_overlay(sel) if tl._ocr else (None, [])
    heat_path = tl._heatmap(sel, feats, words, META)
    st.session_state.update({
        "step": 1,
        "feats": feats,
        "words": words,
        "heat_path": heat_path,
        "ocr_path": ocr_path,
        "scroll_target": "step1"
    })
    st.rerun()

# ----------------------------------------------------------------
#  STEP 1: Attention Heatmap
# ----------------------------------------------------------------
if st.session_state.step >= 1:
    with st.container(border=True):
        st.header("👀 Where does Gen-Z attention land first?")
        # Center and resize heatmap
        cols = st.columns([1,6,1])
        with cols[1]:
            st.markdown(
                f"<div style='text-align:center;'><img src='data:image/png;base64,{to_base64(st.session_state.heat_path)}' width='200'></div>",
                unsafe_allow_html=True
            )
        # Box end
        if st.button("Next ➡️ Features", disabled=st.session_state.step >= 2):
            st.session_state.update({
                "step": 2,
                "scroll_target": "step2"
            })
            st.rerun()

# ----------------------------------------------------------------
#  STEP 2: Visual Features
# ----------------------------------------------------------------
if st.session_state.step >= 2:
    with st.container(border=True):
        st.header("🧬 Visual features extracted")
        feats = st.session_state.feats

        dom_bgr = feats["dominant_color"].astype(int)
        dom_hex = "#%02x%02x%02x" % tuple(dom_bgr[::-1])
        pastel = feats["pastel_palette"]
        glass = feats["has_glass"]
        sym = feats["symmetry"]
        ted = feats["text_edge_density"]
        tar = feats["text_area_ratio"] * 100

        st.markdown(f"""
    <table style="width:100%;border-collapse:collapse">
    <tr><th align=left>Feature</th><th align=left>Value</th><th align=left>Why it matters</th></tr>
    <tr><td><b>Dominant colour</b></td>
        <td><span style="background:{dom_hex};display:inline-block;
            width:30px;height:15px;border:1px solid #666"></span>&nbsp;{dom_hex}</td>
        <td>Soft pastels score&nbsp;<b>+2 Aesthetic</b></td></tr>
    <tr><td><b>Pastel palette</b></td><td>{pastel}</td><td>Pastel look = IG-ready</td></tr>
    <tr><td><b>Detected glass</b></td><td>{glass}</td><td>Glass boosts <b>+3 Luxury</b></td></tr>
    <tr><td><b>Symmetry</b></td><td>{sym:.2f}</td><td>&gt;0.8 looks premium</td></tr>
    <tr><td><b>Edge density</b></td><td>{ted:.3f}</td><td>Busy edges hurt clarity</td></tr>
    <tr><td><b>Text area&nbsp;%</b></td><td>{tar:.2f}%</td><td>Too much text feels noisy</td></tr>
    </table>
    """, unsafe_allow_html=True)

        if st.session_state.ocr_path:
            # Center and resize OCR overlay
            cols = st.columns([1,6,1])
            with cols[1]:
                st.markdown(
                    f"<div style='text-align:center;'><img src='data:image/png;base64,{to_base64(st.session_state.ocr_path)}' width='200'></div>",
                    unsafe_allow_html=True
                )

        if st.button("Next ➡️ Heuristic score", disabled=st.session_state.step >= 3):
            # compute heuristic
            tl = get_tl()
            feats = st.session_state.feats
            words = st.session_state.words
            # call the engine’s heuristic scorer
            scores_obj, contribs = tl._heuristic_score(feats, META, words)
            # get a text recommendation from that heuristic pass
            rec_heur = tl._recommend(scores_obj, feats, META, words)
            # convert to plain dict for display & storage
            sc_dict = scores_obj.as_dict()
            st.session_state.update({
                "step": 3,
                "scores": sc_dict,
                "contribs": contribs,
                "rec_heur": rec_heur,
                "scroll_target": "step3"
            })
            st.rerun()

# ----------------------------------------------------------------
#  STEP 3: Heuristic Score
# ----------------------------------------------------------------
if st.session_state.step >= 3:
    with st.container(border=True):
        st.header("📊 Heuristic perception score")
        sc = st.session_state.scores
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Aesthetic", f"{sc['aesthetic']:.1f}")
        c2.metric("Trust",     f"{sc['trust']:.1f}")
        c3.metric("Luxury",    f"{sc['luxury']:.1f}")
        c4.metric("Purchase",  f"{sc['purchase']:.1f}")

        # show the heuristic recommendation
        st.markdown("**Recommendation (heuristic):**")
        st.info(st.session_state.rec_heur)

        with st.expander("How each rule moved the needle", expanded=True):
            md = "| Rule | Impact | Attributes |\n|------|:--:|-----------|\n"
            for r,amt,tgt in st.session_state.contribs:
                sign = "➕" if amt>0 else "➖"
                md += f"| {r} | {sign}{abs(amt)} | {tgt} |\n"
            st.markdown(md)
        if st.session_state.step == 3 and st.button("Next ➡️ ML score", disabled=st.session_state.step >= 4):
            tl = get_tl()
            dl = tl._ml_delta(st.session_state.selected_image)
            sc_ml = {}
            for k, v in st.session_state.scores.items():
                sc_ml[k] = clamp(v + dl.get(k, 0), 0, 10)
            # new ML‐based recommendation
            rec_ml = tl._recommend(type("S",(),sc_ml)(),
                                st.session_state.feats,
                                META,
                                st.session_state.words)

            st.session_state.update({
                "step": 4,
                "dl": dl,
                "sc_ml": sc_ml,
                "rec_ml": rec_ml,
                "scroll_target": "step4"
            })
            st.rerun()

# ----------------------------------------------------------------
#  STEP 4: ML-Augmented Score
# ----------------------------------------------------------------
if st.session_state.step >= 4:
    with st.container(border=True):
        st.header("🤖 ML-augmented perception score")
        dl    = st.session_state.dl
        sc_ml = st.session_state.sc_ml
        c1,c2,c3,c4 = st.columns(4)
        for col,lbl,key in zip(
            (c1,c2,c3,c4),
            ("Aesthetic","Trust","Luxury","Purchase"),
            ("aesthetic","trust","luxury","purchase")
        ):
            col.metric(lbl, f"{sc_ml[key]:.1f}", delta=f"{dl[key]:+0.2f}")
        
        # show the ML‐based recommendation
        st.markdown("**Recommendation (ML):**")
        st.info(st.session_state.rec_ml)

    if st.session_state.step == 4 and st.button("Next ➡️ Variant generator", disabled=ss_get("step")>=5):
        st.session_state.update({
            "step": 5,
            "scroll_target": "step5"
        })
        st.rerun()

# ----------------------------------------------------------------
#  STEP 5: Variant Generator Options
# ----------------------------------------------------------------
if st.session_state.step >= 5:
    with st.container(border=True):
        st.markdown('<div id="step5"></div>', unsafe_allow_html=True)
        st.header("🎛️  Choose variant generator")
        eng_col, q_col = st.columns(2)
        with eng_col:
            engine  = st.selectbox("Generator engine", ["local","api","openai"],
                                key="variant_engine")
        with q_col:
            quality = st.selectbox("Quality", ["High","Medium","Low"],
                                key="variant_quality")
        if st.button("Generate variants 🎨", disabled=ss_get("step")>=6):
            # generate and save with unique folder & filenames
            variants = []
            out_dir = Path("variants") / datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in get_tl().generate_variants(
                st.session_state.selected_image,
                num=4,
                source=engine,
                quality=quality.lower()
            ):
                img = Image.open(img_path)
                uniq = uuid.uuid4().hex[:8]
                save_path = out_dir / f"variant_{uniq}{Path(img_path).suffix}"
                img.save(save_path)
                variants.append(str(save_path))
            st.session_state.update({
                "step": 6,
                "variants": variants,
                "scroll_target": "step6"
            })
            st.rerun()

# ----------------------------------------------------------------
#  STEP 6: Show Generated Variants
# ----------------------------------------------------------------
if st.session_state.step >= 6:
    with st.container(border=True):
        st.header("🎨 AI-generated variants")
        variants = st.session_state.variants or []
        st.markdown("</div>", unsafe_allow_html=True)
        if variants:
            # show 2 side-by-side, larger
            for i in range(0, len(variants), 2):
                cols = st.columns(2, gap="large")
                for col, vp in zip(cols, variants[i:i+2]):
                    with col:
                        st.image(vp, width=450)
                        with open(vp, "rb") as fh:
                            st.download_button("Download", fh.read(),
                                            file_name=Path(vp).name)

# ----------------------------------------------------------------
#  Fire the scroll if needed
# ----------------------------------------------------------------
do_scroll()

# ----------------------------------------------------------------
#  Footer
# ----------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.caption("© 2025 • GenzView Studio – technical demo")
