# Streamlit app — Body-Fluid Cell Detector

Two modes: **CX33 microscope** and **Smartphone (via eyepiece)**. Each loads the
BFCell-Attn weights trained on that source from Hugging Face.

## Run locally
```bash
pip install -r requirements.txt
export HF_REPO="your-username/bodyfluid-bfcell-attn"   # repo holding the .pt files
streamlit run app.py
```

## Deploy free
- **Streamlit Community Cloud:** push this repo to GitHub, "New app", point at
  `streamlit_app/app.py`, add `HF_REPO` in *Advanced settings ▸ Secrets*.
- **Hugging Face Spaces:** create a Streamlit Space, upload these files, set
  `HF_REPO` as a Space variable.

The `.pt` files are produced by notebook `07` (the "Save best model to Hugging
Face" cell), stored as `CX33/bfcell_attn_best.pt` and
`Smartphone/bfcell_attn_best.pt`.
