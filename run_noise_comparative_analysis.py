import torch
from diffusers import DiffusionPipeline
from setup import move_new_pipeline_to_diffusers

move_new_pipeline_to_diffusers()

device = "cuda" if torch.cuda.is_available() else "mps"
pipe = DiffusionPipeline.from_pretrained(
    "CompVis/ldm-celebahq-256",
    custom_pipeline="latent_diffusion_noise_comparative_analysis",
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)
