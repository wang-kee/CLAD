### CLAD: A Contrastive Learning based Approach for Background Debiasing

This repository contains codes for [CLAD: A Contrastive Learning based Approach for Background Debiasing (BMVC 2022)](https://arxiv.org/abs/2210.02748). 

We propose a contrastive learning framework, CLAD, for reducing the effect of image background in image classification. Through proper design of contrastive samples, CLAD is trained to encourage semantic focus on object foregrounds and penalize using features from the background. Please follow the following steps.

#### Example

Install requirements: ```pip install -r requirements.txt```

##### Step 1: download datasets and imagenet-pretrained weights

We evaluate our method on the ImageNet-9 dataset, and created two foreground sets, with either GrabCut or U2-Net, by segmentation on the original images (for the train set ONLY). The dataset can be downloaded [here](https://drive.google.com/file/d/1FMnN8wd-XmnScV6mwWIeKXXnK0-zgoBL/view?usp=share_link), and then unzipped to the datasets directiory of the repository. Note that if you are using the original ImageNet-9 dataset, we suggest you to generate the foreground segmentations yourself with either GrabCut or U2-Net as we described in the paper rather than directly using the one in ImageNet-9 dataset, as the original ImageNet-9 dataset is saved in JPEG version which lowers the image quality and introduces noise in inpaiting foreground when generating positive samples. 

Besides, please also download ImageNet pretrained weights of a ResNet-50 from [here](https://drive.google.com/file/d/1idTHoHo6lGBdvFRyB1lDCMpug5AtxxJB/view?usp=sharing) and put to the pretrained_weights directiory, otherwise you could also set the argument --train_from_scratch to 1 (hinders performance).

##### Step 2: Train model

* Train baseline:
```python main_clad_bg.py --with_con_loss 0```
* Train CLAD:
```python main_clad_bg.py --with_con_loss 1```
* Train CLAD+:
```python main_clad_bg.py --with_con_loss 1 --with_pos_loss 1```