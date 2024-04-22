from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from transformers import set_seed
import matplotlib.pyplot as plt
from PIL import Image

# Define CFG class
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

# Download the Stable Diffusion model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float32,  # Change to torch.float32
    revision="fp16",
    guidance_scale=CFG.image_gen_guidance_scale
)
image_gen_model = image_gen_model.to(CFG.device)

# Function to generate images
def generate_image(prompt, model):
    image = model(
        prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

# Example usage
resulting_image = generate_image("astronaut riding horse", image_gen_model)

# Display the generated image
plt.imshow(resulting_image)
plt.axis('off')
plt.show()
