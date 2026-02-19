import streamlit as st
import requests
import io
from PIL import Image

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

# ── Freely accessible models (no license gate) ───────────────
MODELS = {
    "✅ Dreamlike Photoreal 2.0 (Best Quality)": "dreamlike-art/dreamlike-photoreal-2.0",
    "✅ Stable Diffusion 2.1 (Fast & Reliable)": "stabilityai/stable-diffusion-2-1",
    "✅ Openjourney v4 (Artistic / MidJourney style)": "prompthero/openjourney-v4",
    "✅ Realistic Vision v3 (Photorealistic)":   "SG161222/Realistic_Vision_V3.0_VAE",
}

BASE_URL = "https://router.huggingface.co/hf-inference/models"


# ── Query Function ────────────────────────────────────────────
def query(hf_token: str, model_id: str, payload: dict):
    api_url = f"{BASE_URL}/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)

        if response.status_code == 401:
            st.error("❌ Invalid token — please double-check your Hugging Face API token.")
            return None
        elif response.status_code == 403:
            st.error(
                "❌ Access denied for this model. Try a different model from the dropdown, "
                "or accept the license at huggingface.co/models"
            )
            return None
        elif response.status_code == 503:
            st.warning("⏳ Model is loading on HF servers. Wait 20–30 sec and try again.")
            return None
        elif response.status_code != 200:
            st.error(f"❌ API Error {response.status_code}: {response.text[:300]}")
            return None

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            st.error(f"❌ Unexpected response (not an image): {response.text[:300]}")
            return None

        return response.content

    except requests.exceptions.Timeout:
        st.error("❌ Request timed out — model may be busy. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error. Please check your internet.")
        return None


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Configuration")
    st.markdown(
        "Get a **free token** at "
        "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)"
    )

    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="Used only for this request. Never stored or logged."
    )

    if hf_token:
        st.success("✅ Token entered")
    else:
        st.info("ℹ️ Enter your token to enable generation")

    st.divider()

    selected_label = st.selectbox("🤖 Choose Model", list(MODELS.keys()))
    selected_model = MODELS[selected_label]
    st.caption(f"`{selected_model}`")

    st.divider()
    st.caption("🔒 Your token is never stored — it goes directly to Hugging Face per request.")


# ── Main UI ───────────────────────────────────────────────────
st.title("🎨 AI Image Generator")
st.caption("Powered by Hugging Face Inference API — no license required models")
st.divider()

prompt = st.text_area(
    "✏️ Describe the image you want:",
    placeholder="A futuristic cyberpunk city at night, neon lights, rain-soaked streets, photorealistic",
    height=100
)

negative_prompt = st.text_input(
    "🚫 Negative prompt (what to avoid):",
    value="blurry, low quality, distorted, watermark, ugly, duplicate, deformed",
)

col1, col2 = st.columns(2)
with col1:
    steps = st.slider("Inference Steps", 10, 50, 30, 5,
                      help="More steps = better quality but slower.")
with col2:
    guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5, 0.5,
                         help="Higher = image follows prompt more strictly.")

st.divider()

# ── Generate ──────────────────────────────────────────────────
if st.button("✨ Generate Image", type="primary", use_container_width=True):

    if not hf_token.strip():
        st.warning("⚠️ Please enter your Hugging Face API token in the sidebar.")
        st.stop()

    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt.")
        st.stop()

    with st.spinner(f"🖼️ Generating with `{selected_model}`... (20–40 sec)"):
        payload = {
            "inputs": prompt.strip(),
            "parameters": {
                "negative_prompt": negative_prompt.strip(),
                "num_inference_steps": steps,
                "guidance_scale": guidance,
            }
        }
        image_bytes = query(hf_token.strip(), selected_model, payload)

    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            st.success("✅ Image generated successfully!")
            st.image(image, caption=f'"{prompt}"', use_container_width=True)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)

            st.download_button(
                label="⬇️ Download Image (PNG)",
                data=buf.getvalue(),
                file_name="generated_image.png",
                mime="image/png",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"❌ Could not render image: {str(e)}")
