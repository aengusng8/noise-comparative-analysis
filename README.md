# noise-comparative-analysis
This repository is to implement and extend the second contribution of [Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227) paper by investigating diverse diffusion models and datasets.
- models: DDPM/DDIM, LDM (latent diffusion model)
- datasets: CelebA-HQ, FFHQ

## Results
Given a input image, we start denoising them at different noise step.
![noise-comparative-analysis](https://user-images.githubusercontent.com/67547213/224677066-4474b2ed-56ab-4c27-87c6-de3c0255eb9c.jpeg)

## How to run
Clone this repository
```
git clone https://github.com/aengusng8/noise-comparative-analysis.git
cd noise-comparative-analysis
```
Then, run below command to install diffusers
```
python setup.py --clone_diffusers --install_editable_diffusers
```
Before running analysis, you can delete my previous result
```
rm -rf ./result
mkdir result
```
Run analysis with below file which you can modify if needed
```
python run_noise_comparative_analysis.py
```

## Acknowledgments
- Thanks to Choi et al for releasing their paper [Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227). 
- This repository is based on [huggingface/diffusers](https://github.com/huggingface/diffusers).
