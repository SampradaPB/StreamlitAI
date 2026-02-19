import streamlit as st
import requests
import io
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="Fast AI Image Gen", page_icon="🎨")

# 2. API Setup
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"  # More stable & actively hosted model
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

# ✅ FIX 1: Removed accidental extra space before the token
HF_TOKEN = "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"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def query(payload):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=120  # ✅ FIX 2: Added timeout to prevent hanging forever
        )

        # ✅ FIX 3: Handle all common status codes clearly
        if response.status_code == 401:
            st.error("❌ Unauthorized: Your Hugging Face token is invalid or expired.")
            return None
        elif response.status_code == 403:
            st.error("❌ Forbidden: You may not have access to this model. Try accepting the model's license on HuggingFace.co")
            return None
        elif response.status_code == 503:
            st.warning("⏳ Model is loading on Hugging Face servers. Please wait 20–30 seconds and try again.")
            return None
        elif response.status_code != 200:
            st.error(f"❌ API Error ({response.status_code}): {response.text}")
            return None

        # ✅ FIX 4: Verify response is actually image bytes, not a JSON error
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            st.error(f"❌ API returned non-image response: {response.text[:300]}")
            return None

        return response.content

    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The model may be overloaded. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error. Please check your internet connection.")
        return None


# 3. UI Layout
st.title("SAM Image Generator 🚀")
st.caption("Powered by Hugging Face Inference API")

prompt = st.text_input(
    "Enter your prompt:",
    placeholder="A futuristic cyberpunk city at night, neon lights, photorealistic"
)

# ✅ FIX 5: Added negative prompt for better quality results
negative_prompt = st.text_input(
    "Negative prompt (optional):",
    placeholder="blurry, low quality, distorted, ugly",
    value="blurry, low quality, distorted, watermark"
)

if st.button("Generate Image ✨", type="primary"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt.")
    else:
        with st.spinner("🎨 Generating your image... this may take 20–40 seconds..."):
            # ✅ FIX 6: Pass parameters properly including negative prompt
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                }
            }
            image_bytes = query(payload)

            if image_bytes:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    st.success("✅ Image generated successfully!")
                    st.image(image, caption=f"'{prompt}'", use_container_width=True)

                    # Download button
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    buf.seek(0)  # ✅ FIX 7: Reset buffer position before reading

                    st.download_button(
                        label="⬇️ Download Image",
                        data=buf.getvalue(),
                        file_name="generated_ai.png",
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"❌ Could not render image: {str(e)}")
                    st.info("The API may have returned an error message instead of an image.")