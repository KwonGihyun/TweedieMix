# TweedieMix

Official source codes for "[TweedieMix: Improving Multi-Concept Fusion for Diffusion-based Image/Video Generation](https://arxiv.org/abs/2410.05591)"

## Environments
```
$ conda create -n tweediemix python=3.11.8
$ conda activate tweediemix
$ pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
$ git clone https://github.com/KwonGihyun/TweedieMix
$ cd TweedieMix/text_segment
$ git clone https://github.com/IDEA-Research/GroundingDINO.git
$ cd GroundingDINO/
$ pip install -e .
```

download and install ```xformers``` which is compatible for your own environment in [https://download.pytorch.org/whl/xformers/](https://download.pytorch.org/whl/xformers/)

e.g. ```pip install xformers-0.0.26+cu118-cp311-cp311-manylinux2014_x86_64.whl```

If you have problem in installing GroundingDINO, please refer to [original repository](https://github.com/IDEA-Research/GroundingDINO)
## Preliminary : Single-concept Training
Train single concept aware model using [Custom Diffusion](https://github.com/adobe-research/custom-diffusion) framework

```
bash singleconcept_train.sh
```

Most of the concept datasets can be downloaded from [customconcept101 dataset](https://github.com/adobe-research/custom-diffusion/blob/main/customconcept101/README.md)

We provide both of *Custom Diffusion* (key,query weight finetuning) and *Low-Rank Adaptation*

We also provide several pre-trained weights in [LINK](https://drive.google.com/drive/folders/1PvNAxDtV4bCIekkI2uMTUE5J6gceDxaU?usp=drive_link)

## Multi Concept Generation
Sampling multi-concept aware images with personalized checkpoints

we provide script for sampling multi-concepts for example,

```
bash sample_catdog.sh
```

we also provide several scripts for multi-concept generation for both of custom diffusion weight or LoRa weights.

For different generation setting, adjust parameters in the bash file.

## Video Generation
After generating multi-concept image, generate video output using Image-to-Video Diffusion

we provide script for I2V generation

```
python run_video.py
```

For different generation setting, adjust parameters in the script file.

## 📋 Results

### Multi-Concept Image Generation Results

   ![multlconcept1](./asset/Fig_supp_1-min.jpg)
   
   ![multiconcept2](./asset/fig_supp_2-min.jpg)

### Multi-Concept Video Generation Results
#### Video Customization Comparison
   ![video_comp](./asset/forgit_video.gif)
#### More Videos
   <img src="./asset/pandateddy_4510.gif" width="400"/> <img src="./asset/pandateddy_5305.gif" width="400"/> 

   <img src="./asset/womancat_3450.gif" width="400"/> <img src="./asset/womancat_4424.gif" width="400"/> 

   <img src="./asset/catdogmountain_732.gif" width="400"/> <img src="./asset/catdogmountain_8960.gif" width="400"/> 

   <img src="./asset/manwoman_4599.gif" width="400"/> <img src="./asset/catdog_4155.gif" width="400"/> 
   
## References
If you find this paper useful for your research, please consider citing

```bib
@InProceedings{
  kwon2024tweedie,
  title={TweedieMix: Improving Multi-Concept Fusion for Diffusion-based Image/Video Generation},
  author={Kwon, Gihyun and Ye, Jong Chul},
  booktitle={https://arxiv.org/abs/2410.05591},
  year={2024}
}
```

Also please refer to our previous version Concept Weaver

```bib
@InProceedings{kwon2024concept,
    author    = {Kwon, Gihyun and Jenni, Simon and Li, Dingzeyu and Lee, Joon-Young and Ye, Jong Chul and Heilbron, Fabian Caba},
    title     = {Concept Weaver: Enabling Multi-Concept Fusion in Text-to-Image Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {8880-8889}
}
```

## Acknowledgement

Our source code is based on [Plug-and-Play Diffusion](https://pnp-diffusion.github.io/) , [Custom Diffusion](https://github.com/adobe-research/custom-diffusion), [LangSAM](https://github.com/luca-medeiros/lang-segment-anything)

