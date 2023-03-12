import torch
from PIL import Image
import argparse
import os


from diffusers import DiffusionPipeline
from setup import move_new_pipeline_to_diffusers

move_new_pipeline_to_diffusers()

CONFIG = {
    "ddim": {
        "model_id": "google/ddpm-ema-celebahq-256",
        "custom_pipeline": "diffusers-0.13.0/examples/community/ddim_noise_comparative_analysis.py",
    },
    "latent_diffusion": {
        "model_id": "CompVis/ldm-celebahq-256",
        "custom_pipeline": "diffusers-0.13.0/examples/community/latent_diffusion_noise_comparative_analysis.py",
    },
}


def main(latent_diffusion=False):
    image_path = "images/CelebA-HQ"
    images = [
        (Image.open(f"{image_path}/{image_name}"), image_name.split(".")[0])
        for image_name in os.listdir(image_path)
    ]
    device = "cuda" if torch.cuda.is_available() else "mps"

    pipeline_name = "latent_diffusion" if latent_diffusion else "ddim"
    print(f"Running {pipeline_name} pipeline on {device}")
    pipe = DiffusionPipeline.from_pretrained(
        CONFIG[pipeline_name]["model_id"],
        custom_pipeline=CONFIG[pipeline_name]["custom_pipeline"],
        # torch_dtype=torch.float16,
    ).to(device)

    strengths = [x / 10.0 for x in range(1, 10, 1)]
    print("strengths", strengths)
    for strength in strengths:
        # TODO: delete this for-loop
        for image in images:
            denoised_image = pipe(image[0], strength=strength).images[0]
            denoised_image.save(f"result/{pipeline_name}_{image[1]}_{strength}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent_diffusion",
        action="store_true",
        default=False,
        help="Use latent diffusion",
    )

    args = parser.parse_args()
    main(**vars(args))
