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
    import cv2, torch
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    tm = model.model.eval()
    device = next(tm.parameters()).device
    rgb = np.float32(np.array(pil_img.convert("RGB").resize((imgsz, imgsz)))) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    layers = [list(tm.model)[-2]]
    cam = EigenCAM(model=tm, target_layers=layers)
    # YOLO returns a tuple, so pass a dummy target to skip BaseCAM's argmax-on-output
    gray = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]
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
    nucleated_only = st.checkbox("% of nucleated cells only (exclude crystals)", value=False)
    st.markdown("---")
    st.markdown("**Mode picks the model trained on that source's images**, so the "
                "colour/optics domain matches your upload.")

up_files = st.file_uploader("Upload one or more body-fluid smear images (select multiple in the dialog)",
                            type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if up_files:
    try:
        model = load_model(hf_repo, WEIGHTS[mode])
    except Exception as e:
        st.error(f"Could not load weights from {hf_repo}. Set a valid HF repo. ({e})")
        st.stop()

    import pandas as pd, collections
    agg = collections.Counter()
    st.subheader(f"Per-image results ({len(up_files)} image(s))")
    for f in up_files:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
        res = model.predict(np.array(img), conf=conf, imgsz=640, verbose=False)[0]
        names = res.names
        img_counts = collections.Counter()
        if res.boxes is not None and len(res.boxes) > 0:
            img_counts = collections.Counter(names[int(c)] for c in res.boxes.cls.cpu().numpy())
            agg.update(img_counts)
        with st.expander(f"{f.name} — {sum(img_counts.values())} cells detected", expanded=True):
            c1, c2 = st.columns(2)
            c1.image(img, caption="input", use_container_width=True)
            c2.image(res.plot()[:, :, ::-1], caption="detections", use_container_width=True)
            if show_xai:
                try:
                    st.image(run_eigencam(model, img), caption="EigenCAM", use_container_width=True)
                except Exception as e:
                    st.warning(f"XAI unavailable: {e}")

    # ---- pooled differential across ALL uploaded images ----
    counts = dict(agg)
    if nucleated_only:
        counts = {k: v for k, v in counts.items() if "crystal" not in k.lower()}
    total = sum(counts.values())
    if total > 0:
        rows = [{"Cell type": k, "Count": v, "%": round(100 * v / total, 1)}
                for k, v in sorted(counts.items(), key=lambda x: -x[1])]
        df = pd.DataFrame(rows)
        st.subheader(f"Pooled differential — {total} cells across {len(up_files)} image(s)")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download differential (CSV)", df.to_csv(index=False),
                           file_name="differential_count.csv", mime="text/csv")
        st.caption("Pooled over all uploaded images at the current confidence threshold. "
                   + ("Nucleated cells only (crystals excluded). " if nucleated_only else "")
                   + "Research/educational use — not a validated clinical diagnostic.")
    else:
        st.info("No cells detected above the confidence threshold.")
else:
    st.info("⬆️ Upload one or more images to begin. Use the sidebar to switch between "
            "CX33 microscope and Smartphone modes.")
