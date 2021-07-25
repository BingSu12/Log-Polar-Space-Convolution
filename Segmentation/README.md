# Log-Polar Space Convolution
Segmentation using DeepLabV3Plus with Log-Polar Space Convolution (LPSC) on the VOC2012 dataset. MobileNet is used as the backbone.



# Acknowledgments
We adapted the code of a PyTorch implementation of DeepLabV3Plus which is publicly available at https://github.com/VainF/DeepLabV3Plus-Pytorch.
We replaced some conventional convolutions with our LPSC in the ASPP model. Please also check the license and usage there if you want to make use of this code. 

The usage of training and testing of the DeepLabV3Plus-LPSC model remains unchanged with DeepLabV3. For convenience, we copied the instructions related to the VOC2012 dataset from the Readme file of the DeepLabV3Plus PyTorch implementation https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/README.md as follows. 

###############################################################################
## Quick Start

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

#### Pascal VOC
You can run train.py with "--download" option to download and extract pascal voc dataset. The defaut path is './datasets/data':

```
/datasets
    /data
        /VOCdevkit 
            /VOC2012 
                /SegmentationClass
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

#### Pascal VOC trainaug

*./datasets/data/train_aug.txt* includes names of 10582 trainaug images (val images are excluded). You need to download additional labels from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). Those labels come from [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet).

Extract trainaug labels (SegmentationClassAug) to the VOC2012 directory.

```
/datasets
    /data
        /VOCdevkit  
            /VOC2012
                /SegmentationClass
                /SegmentationClassAug  # <= the trainaug labels
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

### 3. Train on Pascal VOC2012 Aug

#### Visualize training (Optional)

Start visdom sever for visualization. Please remove '--enable_vis' if visualization is not needed. We did not use visualization.

```bash
# Run visdom server on port 28333
visdom -port 28333
```

#### Train with OS=16

Run main.py with *"--year 2012_aug"* to train your model on Pascal VOC2012 Aug. You can also parallel your training on 4 GPUs with '--gpu_id 0,1,2,3'

**Note: There is no SyncBN in this repo, so training with multple GPUs may degrades the performance.

```bash
python main.py --model deeplabv3plus_mobilenet --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
```

With visualization
```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
```

#### Continue training

Run main.py with '--continue_training' to restore the state_dict of optimizer and scheduler from YOUR_CKPT.

```bash
python main.py ... --ckpt YOUR_CKPT --continue_training
```

### 4. Test

Results will be saved at ./results.

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --test_only --save_val_results
```