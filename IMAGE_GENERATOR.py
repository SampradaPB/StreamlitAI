import torch
import customtkinter as ctk
from PIL import Image, ImageTk
from diffusers import StableDiffusionPipeline
import threading

# 1. System Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SamImageGenerator(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SAM Image Generator")
        self.geometry("600x800")

        # Loading the Model (Optimized for 8GB GPU)
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # We load in float16 to save 50% VRAM without losing noticeable quality
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # UI Elements
        self.header = ctk.CTkLabel(
            self, 
            text="SAM Image Generator", 
            font=ctk.CTkFont(family="Helvetica", size=32, weight="bold")
        )
        self.header.pack(pady=30)

        self.prompt_entry = ctk.CTkEntry(
            self, 
            placeholder_text="Enter the prompt", 
            width=500, 
            height=45,
            font=("Arial", 16)
        )
        self.prompt_entry.pack(pady=10)

        self.gen_button = ctk.CTkButton(
            self, 
            text="Generate Image", 
            command=self.start_generation, 
            font=("Arial", 16, "bold"),
            height=45
        )
        self.gen_button.pack(pady=20)

        # Placeholder for the image
        self.image_display = ctk.CTkLabel(self, text="", width=512, height=512)
        self.image_display.pack(pady=20)

        self.status_label = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.status_label.pack(pady=5)

    def start_generation(self):
        # Run generation in a separate thread so the UI doesn't freeze
        prompt = self.prompt_entry.get()
        if not prompt:
            self.status_label.configure(text="Please enter a prompt first!", text_color="red")
            return
            
        self.gen_button.configure(state="disabled", text="Generating...")
        self.status_label.configure(text="Processing AI model...", text_color="yellow")
        
        thread = threading.Thread(target=self.generate_image, args=(prompt,))
        thread.start()

    def generate_image(self, prompt):
        try:
            # Inference (num_inference_steps=30 is a good balance for 8GB VRAM speed)
            with torch.autocast("cuda"):
                image = self.pipe(prompt, num_inference_steps=30).images[0]
            
            # Save and Update UI
            image.save("generated.png")
            
            # Convert for Tkinter
            img_ctk = ctk.CTkImage(light_image=image, dark_image=image, size=(450, 450))
            
            self.image_display.configure(image=img_ctk)
            self.status_label.configure(text="Generation Complete!", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}", text_color="red")
        finally:
            self.gen_button.configure(state="normal", text="Generate Image")

if __name__ == "__main__":
    app = SamImageGenerator()
    app.mainloop()