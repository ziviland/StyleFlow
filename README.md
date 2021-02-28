# Editor to change StyleGAN2 images manipulating latent W vector. Based on StyleFlow and GANSpace frameworks.
![Python 3.7](https://img.shields.io/badge/Python-3.7-green.svg?style=plastic)
![pytorch 1.1.0](https://img.shields.io/badge/Pytorch-1.1.0-green.svg?style=plastic)
![TensorFlow 1.15.0](https://img.shields.io/badge/TensorFlow-1.15.0-green.svg?style=plastic)
![Torchdiffeq 0.0.1](https://img.shields.io/badge/Torchdiffeq-0.0.1-green.svg?style=plastic)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziviland/styleflow_ganspace_latent_editor/blob/master/Latent_vector_editor_using_StyleFlow_and_GANSpace.ipynb)

![Sequantial edit of latent space](https://raw.githubusercontent.com/ziviland/styleflow_ganspace_latent_editor/master/teaser.png)

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
git clone https://github.com/ziviland/styleflow_ganspace_latent_editor.git
cd styleflow_ganspace_latent_editor/
```

This code requires PyTorch(for CNF), TensorFlow(for StyleGAN2), Torchdiffeq, and Python 3+ Please install dependencies by
```bash
pip install -r requirements.txt
```

This version of StyleGAN2 relies on TensorFlow 1.x.

## Installation (Docker)

Didn't test, but may work. 

Clone this repo.

```bash
git clone https://github.com/ziviland/styleflow_ganspace_latent_editor.git
cd styleflow_ganspace_latent_editor/
```

You must have CUDA (>=10.0 && <11.0) and [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) installed first !

Then, run :

```bash
xhost +local:docker # Letting Docker access X server
wget -P stylegan/ http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl
docker-compose up --build # Expect some time before UI appears
```

When finished, run :

```bash
xhost -local:docker
```

## License

Licebse according to [StyleFlow](https://github.com/RameenAbdal/StyleFlow)([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)) and [GANSpace](https://github.com/harskish/ganspace)([Apache License 2.0]https://github.com/harskish/ganspace#license) repositories

## Acknowledgments
This repository is heavily based on [StyleFlow](https://github.com/RameenAbdal/StyleFlow) and [GANSpace](https://github.com/harskish/ganspace) frameworks.

StyleFlow implementation builds upon the awesome work done by Karras et al. ([StyleGAN2](https://github.com/NVlabs/stylegan2)), Chen et al. ([torchdiffeq](https://github.com/rtqichen/torchdiffeq)) and Yang et al. ([PointFlow](https://arxiv.org/abs/1906.12320)).

