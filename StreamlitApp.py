import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="SAM Image Generator", page_icon="🎨")

# 2. Model Loading (Cached so it doesn't reload on every button click)
@st.cache_resource
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Using float16 for GPU to save VRAM
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype
    ).to(device)
    return pipe, device

# Initialize the model
pipe, device = load_pipeline()

# 3. UI Layout
st.title("SAM Image Generator 🚀")
st.markdown("Enter a prompt below to generate an image using Stable Diffusion v1.5.")

# Sidebar for settings (equivalent to your optimization tweaks)
with st.sidebar:
    st.header("Settings")
    steps = st.slider("Inference Steps", min_value=10, max_value=100, value=30)
    guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)

# Main Input
prompt = st.text_input("What do you want to create?", placeholder="e.g., A futuristic city in the clouds")

# 4. Generation Logic
if st.button("Generate Image", type="primary"):
    if not prompt:
        st.error("Please enter a prompt first!")
    else:
        with st.spinner("Processing AI model... This may take a moment."):
            try:
                # Inference
                if device == "cuda":
                    with torch.autocast("cuda"):
                        image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale).images[0]
                else:
                    image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale).images[0]

                # Display Result
                st.image(image, caption=f"Generated: {prompt}", use_container_width=True)
                
                # Download Button
                st.download_button(
                    label="Download Image",
                    data=open("generated.png", "rb") if image.save("generated.png") else image.save("generated.png") or open("generated.png", "rb"),
                    file_name="generated_image.png",
                    mime="image/png"
                )
                st.success("Generation Complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")