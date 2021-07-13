### SPL：Context-Aware Image Inpainting with Learned Semantic Priors
Code for Context-Aware Image Inpainting with Learned Semantic Priors, IJCAI 2021


So, in the following days, I will complete this project step by step. There may be many changes until the completion, but I hope I can provide a detailed introduction about our work, especially for the dataset processing part and evaluation part, which are important for the beginners. Thank you for all of your interest!

[Paper on ArXiv](https://arxiv.org/abs/2106.07220)
### Introduction:
We introduce pretext tasks that are semantically meaningful to estimating the missing contents. In particular, we perform knowledge distillation on pretext models and adapt the features to image inpainting. The learned semantic priors ought to be partially invariant between the high-level pretext task and low-level image inpainting, which not only help to understand the global context but also provide structural guidance for the restoration of local textures. Based on the semantic priors, we further propose a context-aware image inpainting model, which adaptively integrates global semantics and local features in a unified image generator. The semantic learner and the image generator are trained in an end-to-end manner. More details can be found in our [paper](https://arxiv.org/abs/2106.07220).
<p align='center'>  
  <img src='https://github.com/WendongZh/SPL-Image-Inpainting-with-Semantic-Priors/blob/main/img/results.PNG' width='870'/>
</p>

## Prerequisites
- Python 3.7
- PyTorch 1.8 (1.6+ may also work)
- NVIDIA GPU + CUDA cuDNN
- [Inplace_Abn](https://github.com/mapillary/inplace_abn) (only for training with [ASL_TRresNet](https://github.com/Alibaba-MIIL/ASL) model)
- torchlight (We only use it to record the printed information. You can change it as you want.)

## Datasets
### 1) Images
We use [Places2](http://places2.csail.mit.edu), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. To train a model on the full dataset, download datasets from official websites. 

After downloading, run [`flist.py`](flist.py) to generate train and test set file lists. For example, to generate the training set file list on Places2 dataset run:
```bash
mkdir datasets
python flist.py --path path_to_places2_train_set --output ./datasets/places_train.flist
```
1) About image size. For all datasets except the celeba dataset, we resize the images to 256x256 for both training and testing. For celeba dataset, we first perform center crop and then also resize the images to 256x256.
2) About dataset usage. For Places2 dataset, we use the Places365-Standard subset. More specially, we use the **"Small images (256 * 256) with easy directory structure"** part to train and evaluate our model. For Celeba dataset, we use the train and test sets from **"img_align_celeba.zip"** file. For Paris StreetView dataset, you need to write an email to [Prof.Pathak](https://www.cs.cmu.edu/~dpathak/) to get the dataset. We **STRONGLY** suggest you first trying your idea on Paris dataset to save your time.
3) About the evaluation. We use the val part of places365-standard (36500), test part of Celeba (about 20000) and val part of Paris (100) to evaluate our method. Besides, Yes, there is no standard image-mask paris for comparisons between different methods. To compare with other methods, as we mentioned in our paper, "We first randomly sample three different mask sets from the whole irregular mask dataset. These mask sets are assigned to different image datasets to form the mask-image pairs. For each image dataset, the mask-image mappings are held for different methods to obtain fair comparisons." In this case, you need first obatin pretrained models of other methods for fair comparisons.

4) About data argumentation. We only use image flip for data argumentation of both masks and images and we do not observe significant improvements when more complicated argumentation methods are applied.

### 2) Irregular Masks
We use the irregular mask dataset provided by [Liu et al.](https://arxiv.org/abs/1804.07723). You can download publically available Irregular Mask Dataset from [their website](https://nv-adlr.github.io/publication/partialconv-inpainting).
1) About dataset usage. We only use the testing mask set (12000) to train and test our model. You can perform more complicatd argumentation methods for masks, such as rotation, translation or randomcrop as suggested by [EC](https://github.com/knazeri/edge-connect).
2) About test mask selection. In our case, we use [`flist.py`](flist.py) to generate mask file lists as explained above. Then, we generated random selected mask index files using numpy for our evaluation. Such as:
```bash
python
import numpy as np
index = np.random.choice(12000, 100) # for paris
np.save('index_paris.npy', index)
```

Alternatively, you can download [Quick Draw Irregular Mask Dataset](https://github.com/karfly/qd-imd) by Karim Iskakov which is combination of 50 million strokes drawn by human hand.

## Getting Strated

Make another directory, e.g ./pretrained_ASL, and download the weights of [TResNet_L](https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md) pretrained on OpenImage dataset to this directory.

Install torchlight
```bash
cd ./torchlight
python setup.py install
```
### 1) Training
We use the DistributedDataParallel (DDP) to train our model, which means that for now, you need at least two GPU cards for training our model. To train our model with only one GPU, you need modify the initialization, datasetloader and optimization parts and I will provide a new version in the future. 

Take the Paris dataset for example, the training commond is as follows:  
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
        --bs 4 --gpus 2 --prefix SPL_paris --with_test \
        --img_flist your/train/flist/of/paris --mask_flist your/flist/of/mask --test_img_flist your/test/flist/of/paris \
        --test_mask_flist your/flist/of/masks --test_mask_index your/npy/file/to/form/img-mask/pairs \
        --dataset paris  --TRresNet_path path/of/ASL/weight --nEpochs 70
```
If you want to retrain your model, you need add 
```bash
--pretrained True --pretrained_sr checkpoints/of/your/model --start_epoch 4
```
During our training stage, we --test_img_flist and --test_mask_index to evaluate the performance of current model. You can change the evaluation number with parameter --val_prob_num. 

For Paris dataset, we train our model for 70 epochs and we deacy the learning rete at about 50 epochs with 0.1. Besides, in the last 10 epochs we remove the prior reconstruction loss as we find this can further improve the performance. For Celeba and Places2 dataset, we only deacy the learning rate at about 30 epochs and train our model for futher 10 epochs.

In our experiments, we usually obtain the best model in the last 4 epochs.

### 2) Test and Evaluation
