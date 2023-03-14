# noise-comparative-analysis

### **Question: What visual concepts do the diffusion models learn from each noise level during training?**

The [P2 weighting (CVPR 2022)](https://arxiv.org/abs/2204.00227) paper proposed an approach to answer this question, which is their second contribution. According to the paper:
- When SNR is high, the model will learn only imperceptible details by solving recovery tasks.
- When SNR is low, the model will learn perceptually rich contents by solving recovery tasks.

This repository aims to implement and extend their second contribution by investigating diverse diffusion models and datasets. The experiments include:
- Models: DDPM/DDIM/PNDM, LDM (Latent Diffusion Model)
- Datasets: CelebA-HQ, FFHQ

In my experiments, I observed that the behavior of the diffusion models I used (DDPM/DDIM/PNDM) was consistent with the [openai/guided-diffusion](https://github.com/openai/guided-diffusion) model used by the authors in the paper. This consistency further supports the authors' aforementioned assumption regarding the visual concepts learned by diffusion models at different noise levels.

## Methodology
To investigate the behavior of the models, we adopt the approach used by the authors in the P2 weighting paper. The approach consists of the following steps:
1. The input is an image x0.
2. We first perturb it to xt using a diffusion process q(xt|x0).
3. We then reconstruct the image with the learned denoising process pθ(ˆx0|xt).
4. We compare x0 and ˆx0 among various t to show how each step contributes to the sample.

## Results
They used [openai/guided-diffusion](https://github.com/openai/guided-diffusion) model to denoise images in FFHQ dataset.
![image](https://user-images.githubusercontent.com/67547213/224747998-d526229f-b20f-49e2-aa7d-6e692b1cd28d.png)

Here is the result of my experiment with DDIM and CelebA-HQ dataset.
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
