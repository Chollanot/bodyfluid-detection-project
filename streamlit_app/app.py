"""
Body-Fluid Cell Detector — Streamlit app
Two modes: CX33 (microscope) and Smartphone (via eyepiece). Each mode loads the
matching BFCell-Attn weights from Hugging Face, runs detection on an uploaded
image, and optionally shows an EigenCAM explanation.

Deploy on Streamlit Community Cloud (free) or Hugging Face Spaces.
Weights are pulled at runtime from the HF repo you set in HF_REPO below (or via
the sidebar), so the app stays lightweight.
"""
import io
import os
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Body-Fluid Cell Detector", page_icon="🔬", layout="wide")

# ---- configuration ---------------------------------------------------------
HF_REPO_DEFAULT = os.environ.get("HF_REPO", "your-username/bodyfluid-bfcell-attn")
WEIGHTS = {  # mode -> file inside the HF repo
    "CX33 microscope": "CX33/bfcell_attn_best.pt",
    "Smartphone (via eyepiece)": "Smartphone/bfcell_attn_best.pt",
}


@st.cache_resource(show_spinner="Downloading model weights…")
def load_model(hf_repo: str, weight_path: str):
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    # register the novel CBAMLite module so the custom checkpoint loads
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import novel_modules as nm
        nm.register_bfcell_modules()
    except Exception as e:
        st.warning(f"CBAMLite registration skipped: {e}")
    local = hf_hub_download(repo_id=hf_repo, filename=weight_path)
    return YOLO(local)


def run_eigencam(model, pil_img, imgsz=640):
    import cv2
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import torch
    rgb = np.float32(np.array(pil_img.convert("RGB").resize((imgsz, imgsz)))) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    layers = [list(model.model.model)[-2]]
    cam = EigenCAM(model=model.model, target_layers=layers)
    gray = cam(input_tensor=tensor)[0]
    return show_cam_on_image(rgb, gray, use_rgb=True)


# ---- UI --------------------------------------------------------------------
st.title("🔬 Body-Fluid Cell Detector")
st.caption("Attention-augmented, imbalance-aware detector (BFCell-Attn). "
           "Choose the capture source that matches your image.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Capture source (mode)", list(WEIGHTS.keys()))
    hf_repo = st.text_input("Hugging Face weights repo", HF_REPO_DEFAULT)
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    show_xai = st.checkbox("Show XAI (EigenCAM) explanation", value=False)
    st.markdown("---")
    st.markdown("**Mode picks the model trained on that source's images**, so the "
                "colour/optics domain matches your upload.")

up = st.file_uploader("Upload a body-fluid smear image", type=["jpg", "jpeg", "png"])

if up:
    img = Image.open(io.BytesIO(up.read())).convert("RGB")
    col1, col2 = st.columns(2)
    col1.subheader("Input"); col1.image(img, use_container_width=True)

    try:
        model = load_model(hf_repo, WEIGHTS[mode])
    except Exception as e:
        st.error(f"Could not load weights from {hf_repo}. Set a valid HF repo. ({e})")
        st.stop()

    res = model.predict(np.array(img), conf=conf, imgsz=640, verbose=False)[0]
    plotted = res.plot()[:, :, ::-1]  # BGR->RGB
    col2.subheader("Detections"); col2.image(plotted, use_container_width=True)

    # detection summary table
    if res.boxes is not None and len(res.boxes) > 0:
        import pandas as pd, collections
        names = res.names
        counts = collections.Counter(names[int(c)] for c in res.boxes.cls.cpu().numpy())
        df = pd.DataFrame(sorted(counts.items(), key=lambda x: -x[1]),
                          columns=["Cell type", "Count"])
        st.subheader("Cell counts")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No cells detected above the confidence threshold.")

    if show_xai:
        st.subheader("XAI — EigenCAM (where the model looked)")
        try:
            st.image(run_eigencam(model, img), use_container_width=True)
        except Exception as e:
            st.warning(f"XAI unavailable: {e}")
else:
    st.info("⬆️ Upload an image to begin. Use the sidebar to switch between "
            "CX33 microscope and Smartphone modes.")
