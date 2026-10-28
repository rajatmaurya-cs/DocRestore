"""
DocRestore AI - Streamlit Web Application
DocRestore AI: Document Shadow Removal Using Pix2Pix GAN

Provides a clean, modern, intuitive Canva-inspired interface for restoring shadowed document images:
- Canva-style centered upload dropzone with custom upload icon.
- Light, elegant color palette (no harsh dark blocks or jarring high-contrast pills).
- Automatically uses Pix2Pix U-Net Generator with multi-channel color ink preservation.
- Side-by-side high-resolution comparison and instant PNG download.
"""

from io import BytesIO
import importlib
from pathlib import Path
from PIL import Image
import streamlit as st

import config
import model
import inference

# Auto-reload submodules on file update
importlib.reload(model)
importlib.reload(inference)

from config import (
    FALLBACK_CHECKPOINT_PATH,
    GENERATOR_CHECKPOINT_PATH,
    get_device,
)
from inference import open_document_image, restore_document_image
from model import is_checkpoint_available, load_generator_checkpoint

# ---------------------------------------------------------
# Page Setup & Modern Light Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="DocRestore AI - Document Shadow Removal",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom High-End Canva-Inspired Pure Dark Mode CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean Deep Black App Background */
    [data-testid="stAppViewContainer"], .stApp {
        background: #0B0F19 !important;
        color: #F8FAFC !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Main Container Spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Refined Dark Header Container */
    .hero-container {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 9999px;
        padding: 0.35rem 0.95rem;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        letter-spacing: 0.02em;
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.03em;
        margin-bottom: 0.6rem;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        line-height: 1.6;
        margin: 0 auto 1.4rem auto;
        max-width: 720px;
    }

    .badge-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    /* Soft, low-contrast dark pills */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255, 255, 255, 0.05);
        color: #CBD5E1;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 9999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .pill-badge-indigo {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border-color: rgba(99, 102, 241, 0.3);
    }

    .pill-badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border-color: rgba(16, 185, 129, 0.3);
    }

    .pill-badge-sky {
        background: rgba(14, 165, 233, 0.15);
        color: #7DD3FC;
        border-color: rgba(14, 165, 233, 0.3);
    }

    /* Canva-Style Single Integrated Upload Dropzone */
    [data-testid="stFileUploader"] {
        margin: 0 auto !important;
        max-width: 100% !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #111827 !important;
        border: 2px dashed rgba(255, 255, 255, 0.18) !important;
        border-radius: 20px !important;
        padding: 3rem 2rem 2.8rem 2rem !important;
        text-align: center !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.25s ease-in-out !important;
        cursor: pointer !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        position: relative !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6366F1 !important;
        background: #1E1B4B !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.25) !important;
        transform: translateY(-2px) !important;
    }

    /* Inject Canva Cloud Upload Icon directly into Streamlit's Dropzone */
    [data-testid="stFileUploaderDropzone"]::before {
        content: "" !important;
        display: block !important;
        width: 72px !important;
        height: 72px !important;
        margin: 0 auto 1.2rem auto !important;
        border-radius: 50% !important;
        background-color: rgba(99, 102, 241, 0.18) !important;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="%23818CF8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/></svg>') !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25) !important;
        transition: transform 0.25s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover::before {
        transform: scale(1.08) translateY(-2px) !important;
    }

    /* Dropzone Main Heading / Button */
    [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"],
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2.2rem !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        margin: 0.6rem 0 !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
    }

    /* Dropzone text instructions */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #CBD5E1 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
    }

    [data-testid="stFileUploaderDropzone"] small {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        margin-top: 0.4rem !important;
    }

    /* Centered Action Button (Canva Indigo Gradient) */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 2.2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(79, 70, 229, 0.5) !important;
    }

    /* Download Button (Emerald Gradient) */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.7rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
        width: 100% !important;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
    }

    /* Comparison Cards */
    .image-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .card-header-tag {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.55rem 0.9rem;
        border-radius: 10px;
        margin-bottom: 0.9rem;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .tag-degraded {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    .tag-restored {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    /* Expander Styling */
    [data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }

    /* Metrics Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        color: #94A3B8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_generator(checkpoint_path_str: str, file_mtime: float):
    """Caches loaded PyTorch model in memory, auto-reloading if weights file updates."""
    device = get_device()
    return load_generator_checkpoint(checkpoint_path=checkpoint_path_str, device=device)


# ---------------------------------------------------------
# Sidebar: System Status & Specifications
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Status")

    device = get_device()
    is_gpu = device.type in ("mps", "cuda")
    device_name = "Apple Silicon MPS (GPU)" if device.type == "mps" else ("CUDA (GPU)" if device.type == "cuda" else "CPU")

    has_model, resolved_path = is_checkpoint_available()

    if has_model:
        st.success(f"**Generator Active**\n\n`{resolved_path.name}`")
    else:
        st.error("**Checkpoint Pending**\n\nPlace `generator.pth` in `models/`")

    st.markdown(
        f"""
        <div style="background: #F1F5F9; padding: 0.9rem 1rem; border-radius: 10px; border: 1px solid #E2E8F0; margin-top: 0.6rem;">
            <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 0.2rem; font-weight: 600;">Hardware Engine</div>
            <div style="font-weight: 700; color: #4F46E5; font-size: 0.92rem;">{device_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🏛️ Model Specifications")
    st.markdown(
        """
        - **Architecture:** Pix2Pix cGAN
        - **Generator:** Symmetrical 8-stage U-Net
        - **Discriminator:** PatchGAN (70×70)
        - **Resolution:** 256×256 with Aspect Unpad
        - **Text Fidelity:** Color-Preserving Retinex
        """
    )
    st.markdown("---")
    st.caption("DocRestore AI • College Deep Learning Project")


# ---------------------------------------------------------
# Modern Refined Dark Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">✨ Deep Learning Document Restoration</div>
        <div class="hero-title">DocRestore AI</div>
        <div class="hero-subtitle">
            Instantly eradicate harsh cast shadows from document photographs while preserving 100% of authentic ink colors, signatures, and handwriting.
        </div>
        <div class="badge-row">
            <span class="pill-badge pill-badge-indigo">🧠 Pix2Pix cGAN</span>
            <span class="pill-badge pill-badge-emerald">✍️ Blue & Color Ink Preserved</span>
            <span class="pill-badge pill-badge-sky">🛡️ Zero Halo Retinex Engine</span>
            <span class="pill-badge">⚡ Sub-Second GPU Inference</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Missing Checkpoint Alert Banner (if applicable)
# ---------------------------------------------------------
if not has_model:
    st.warning(
        """
        ### ⚠️ Trained Model Checkpoint Required
        The application is ready, but **`models/generator.pth`** was not found on disk.
        
        **How to integrate your model:**
        1. Place your trained weights file into: `models/generator.pth`
        2. Refresh this browser tab to unlock deep learning restoration!
        """
    )

# ---------------------------------------------------------
# Canva-Style Centered Upload Area
# ---------------------------------------------------------
col_up_l, col_up_c, col_up_r = st.columns([1, 3.2, 1])

with col_up_c:
    uploaded_file = st.file_uploader(
        "Upload Shadowed Document",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Upload an RGB document image degraded by shadows (JPG, JPEG, PNG).",
    )

if uploaded_file is not None:
    try:
        input_image = open_document_image(uploaded_file)
        w, h = input_image.size
    except Exception as e:
        st.error(f"Error opening image file: {e}")
        st.stop()

    # Centered File Info & Action Bar
    st.markdown(
        f"""
        <div style="text-align: center; margin: 1rem 0 1.2rem 0;">
            <span style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.12); padding: 0.45rem 1.1rem; border-radius: 9999px; font-size: 0.85rem; color: #CBD5E1; font-weight: 600; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                📄 <b>{uploaded_file.name}</b> &nbsp;•&nbsp; {w} × {h} px &nbsp;•&nbsp; RGB
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_model:
        st.info("Inference will unlock as soon as `models/generator.pth` is placed in `models/`.")
    else:
        BEST_SHADOW_INTENSITY = 1.35
        ENHANCE_CONTRAST = True

        col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
        with col_btn_c:
            restore_btn = st.button("✨ Remove Document Shadow", type="primary")

        if restore_btn:
            with st.spinner("Restoring document through Pix2Pix Generator & Color Engine..."):
                try:
                    mtime = Path(resolved_path).stat().st_mtime
                    generator = get_cached_generator(str(resolved_path), mtime)
                    result = restore_document_image(
                        image_input=input_image,
                        generator=generator,
                        device=device,
                        mode="letterbox",
                        preserve_text_details=True,
                        shadow_intensity=BEST_SHADOW_INTENSITY,
                        enhance_text_contrast=ENHANCE_CONTRAST,
                    )
                    st.session_state["restored_img"] = result["restored_image"]
                    st.session_state["latency"] = result["inference_time_seconds"]
                except Exception as err:
                    st.error(f"Inference failed: {err}")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Side-by-Side Comparison Columns
    # ---------------------------------------------------------
    col_input, col_restored = st.columns(2, gap="large")

    with col_input:
        st.markdown(
            """
            <div class="card-header-tag tag-degraded">
                <span>📷 Degraded Input Document</span>
                <span style="font-size: 0.75rem; font-weight: 600;">Original Photo</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(input_image, use_container_width=True)

    with col_restored:
        st.markdown(
            """
            <div class="card-header-tag tag-restored">
                <span>✨ Restored Clean Output</span>
                <span style="font-size: 0.75rem; font-weight: 600;">Shadow-Free</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "restored_img" in st.session_state:
            st.image(st.session_state["restored_img"], use_container_width=True)
        else:
            st.markdown(
                """
                <div style="border: 2px dashed rgba(255, 255, 255, 0.15); border-radius: 14px; background: #111827; padding: 4rem 1rem; text-align: center; color: #94A3B8; margin-top: 0.5rem;">
                    <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">📄</div>
                    <div style="font-weight: 700; color: #F8FAFC; font-size: 1.05rem;">Restored Document Preview</div>
                    <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 0.3rem;">Click <b>✨ Remove Document Shadow</b> above to run</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Status Pill & Download Button
    if "restored_img" in st.session_state:
        restored_img = st.session_state["restored_img"]
        latency = st.session_state.get("latency", 0.0)

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); padding: 0.9rem 1.4rem; border-radius: 12px; margin-top: 1.8rem; margin-bottom: 1.2rem; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
                <span style="color: #34D399; font-weight: 700; font-size: 0.95rem;">✓ Document Shadow Removed • Authentic Ink Colors Preserved</span>
                <span style="color: #94A3B8; font-size: 0.88rem;">Inference Latency: <b style="color: #F8FAFC;">{latency:.2f}s</b> ({device_name})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_dl_l, col_dl_c, col_dl_r = st.columns([1, 2, 1])
        with col_dl_c:
            buf = BytesIO()
            restored_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="⬇️ Download Restored Document (PNG)",
                data=byte_im,
                file_name="docrestore_clean_restored.png",
                mime="image/png",
            )

# ---------------------------------------------------------
# Training History & Loss Convergence Panel
# ---------------------------------------------------------
history_csv = Path("checkpoints/history.csv")
if history_csv.is_file():
    import pandas as pd

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    with st.expander("📊 View Model Training Metrics & Loss Convergence Curves", expanded=False):
        try:
            df_hist = pd.read_csv(history_csv)
            st.markdown("#### Google Colab Training Progression (80 Epochs)")
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Epochs Completed", len(df_hist))
            m_col2.metric("Final Generator Loss", f"{df_hist['G_loss'].iloc[-1]:.4f}")
            m_col3.metric("Final Discriminator Loss", f"{df_hist['D_loss'].iloc[-1]:.4f}")
            m_col4.metric("Best Val L1 Pixel Error", f"{df_hist['val_L1'].min():.4f}")

            st.dataframe(df_hist, use_container_width=True)
            st.line_chart(df_hist.set_index("epoch")[["G_loss", "D_loss", "val_L1"]])

            loss_curve_img = Path("checkpoints/loss_curves.png")
            if loss_curve_img.is_file():
                st.image(str(loss_curve_img), caption="80-Epoch Convergence Plot", use_container_width=True)
        except Exception as e:
            st.caption(f"Could not load training history: {e}")
