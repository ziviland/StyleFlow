# Editor to change StyleGAN2 images manipulating latent W vector. Based on StyleFlow and GANSpace frameworks.
![Python 3.7](https://img.shields.io/badge/Python-3.7-green.svg?style=plastic)
![pytorch 1.1.0](https://img.shields.io/badge/Pytorch-1.1.0-green.svg?style=plastic)
![TensorFlow 1.15.0](https://img.shields.io/badge/TensorFlow-1.15.0-green.svg?style=plastic)
![Torchdiffeq 0.0.1](https://img.shields.io/badge/Torchdiffeq-0.0.1-green.svg?style=plastic)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/ziviland/0a59ee8110adfb5f5cf92aee3cb2e015/stylegan2_latent_editor.ipynb)

![teaser](https://raw.githubusercontent.com/ziviland/stylegan2_latent_editor/master/teaser.png)

This repository is heavily based on [StyleFlow](https://github.com/RameenAbdal/StyleFlow) and [GANSpace](https://github.com/harskish/ganspace) repositories.
Actially, it just combines two of them.

StyleFlow provides very disentangled attribute change, while GANSpace offer opportunity to discover new attributes without additional training.
For details of how they work, please, refer to corresponding papers.

**Available attributes from StyleFlow (based on Continous Normalizing Flows(CNF)):**
* gender
* glasses
* head yaw
* head pitch
* baldness
* beard
* age
* face expression (smile)

There's also available light attributes, but they don't seem impressive to me.

**Available GANSpace attributes (based on PCA components):**
* also baldness to compare
* hair color
* eyes size (change nationality east asian - european)
* eyes openness
* eyebrow thickness
* lipstick and makeup
* open mouth
* skin tone (more like with/without tan)

## Installation

Clone this repo.
```bash
git clone https://github.com/ziviland/stylegan2_latent_editor.git
cd stylegan2_latent_editor/
```

This code requires PyTorch(for CNF), TensorFlow(for StyleGAN2), Torchdiffeq, and Python 3+ Please install dependencies by
```bash
pip install -r requirements.txt
```

This version of StyleGAN2 relies on TensorFlow 1.x.

## Installation (Docker)

Didn't test, but may work. 

```bash
git clone https://github.com/ziviland/stylegan2_latent_editor.git
cd stylegan2_latent_editor/
docker-compose up --build

```
You must have CUDA (>=10.0 && <11.0) and [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) installed first !

## License

License according to [StyleFlow](https://github.com/RameenAbdal/StyleFlow)([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)) and [GANSpace](https://github.com/harskish/ganspace)([Apache License 2.0](https://github.com/harskish/ganspace#license)) repositories

## Acknowledgments
This repository is heavily based on [StyleFlow](https://github.com/RameenAbdal/StyleFlow) and [GANSpace](https://github.com/harskish/ganspace) frameworks.

StyleFlow implementation builds upon the awesome work done by Karras et al. ([StyleGAN2](https://github.com/NVlabs/stylegan2)), Chen et al. ([torchdiffeq](https://github.com/rtqichen/torchdiffeq)) and Yang et al. ([PointFlow](https://arxiv.org/abs/1906.12320)).

