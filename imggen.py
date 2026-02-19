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

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
API_URL  = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"


# ── Query Function (token passed per-request, never stored globally) ──
def query(hf_token: str, payload: dict):
    headers = {"Authorization": f"Bearer {hf_token}"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

        if response.status_code == 401:
            st.error("❌ Invalid token — please double-check your Hugging Face API token.")
            return None
        elif response.status_code == 403:
            st.error("❌ Access denied — you may need to accept the model license on huggingface.co")
            return None
        elif response.status_code == 503:
            st.warning("⏳ Model is still loading on HF servers. Wait 20–30 seconds and try again.")
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
        st.error("❌ Request timed out. The model may be busy — please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error. Please check your internet.")
        return None


# ── Sidebar — Token Entry ─────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Configuration")
    st.markdown(
        "Enter your **Hugging Face API token** below.\n\n"
        "Get one free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)"
    )

    # password=True masks the token — never shown in plain text
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="Your token is used only for this request and is never stored or logged."
    )

    if hf_token:
        st.success("✅ Token entered")
    else:
        st.info("ℹ️ Enter your token to enable generation")

    st.divider()
    st.markdown("**Model:** `stable-diffusion-xl-base-1.0`")
    st.markdown("**Provider:** Hugging Face Inference API")
    st.divider()
    st.caption("🔒 Your token is never stored, logged, or sent anywhere except directly to Hugging Face.")


# ── Main UI ───────────────────────────────────────────────────
st.title("🎨 AI Image Generator")
st.caption("Powered by Stable Diffusion XL via Hugging Face")
st.divider()

prompt = st.text_area(
    "✏️ Describe the image you want:",
    placeholder="A futuristic cyberpunk city at night, neon lights, rain-soaked streets, photorealistic",
    height=100
)

negative_prompt = st.text_input(
    "🚫 Negative prompt (what to avoid):",
    value="blurry, low quality, distorted, watermark, ugly, duplicate",
    help="Describe things you do NOT want in the image."
)

col1, col2 = st.columns(2)
with col1:
    steps = st.slider("Inference Steps", min_value=10, max_value=50, value=30, step=5,
                      help="More steps = better quality but slower.")
with col2:
    guidance = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5,
                         help="Higher = image follows prompt more strictly.")

st.divider()

# ── Generate Button ───────────────────────────────────────────
generate_btn = st.button("✨ Generate Image", type="primary", use_container_width=True)

if generate_btn:
    if not hf_token.strip():
        st.warning("⚠️ Please enter your Hugging Face API token in the sidebar first.")
        st.stop()

    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt describing your image.")
        st.stop()

    with st.spinner("🖼️ Generating your image... this may take 20–40 seconds..."):
        payload = {
            "inputs": prompt.strip(),
            "parameters": {
                "negative_prompt": negative_prompt.strip(),
                "num_inference_steps": steps,
                "guidance_scale": guidance,
            }
        }

        # Token passed directly into function — never stored in session or global state
        image_bytes = query(hf_token.strip(), payload)

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
