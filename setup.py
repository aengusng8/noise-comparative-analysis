import argparse
import os, sys


def setup(clone_diffusers=False, install_editable_diffusers=False):
    if clone_diffusers:
        clone_diffusers_library()

    if install_editable_diffusers:
        install_editable_diffusers_library()

    move_new_pipeline_to_diffusers()


def clone_diffusers_library():
    wget_link = "https://github.com/huggingface/diffusers/archive/refs/tags/v0.13.0.zip"
    os.system(f"wget {wget_link}")
    os.system("unzip v0.13.0.zip")
    os.system("rm v0.13.0.zip")
    print("Successfully clone the stable version of diffusers (v0.13.0)")


def install_editable_diffusers_library():
    os.system("cd diffusers-0.13.0 && pip install -e \".[torch]\" && cd ..")
    print("Successfully install the diffusers library (v0.13.0) in editable mode")


def move_new_pipeline_to_diffusers():
    source2destination = [
        (
            "noise_comparative_analysis/latent_diffusion_noise_comparative_analysis.py",
            "diffusers-0.13.0/examples/community/",
        ),
    ]

    for source, destination in source2destination:
        os.system(f"cp {source} {destination}")

    print("Successfully move the new pipeline to diffusers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clone_diffusers",
        action="store_true",
        default=False,
        help="Clone the stable version of diffusers (v0.13.0)",
    )
    parser.add_argument(
        "--install_editable_diffusers",
        action="store_true",
        default=False,
        help="Install the diffusers library (v0.13.0) in editable mode",
    )
    args = parser.parse_args()

    setup(**vars(args))
