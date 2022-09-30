# Log-Polar Space Convolution
Classification using Log-Polar Space Convolution (LPSC) on CIFAR-10/100 and ImageNet with PyTorch.



# Acknowledgments
We adapted the code of a PyTorch implementation of image classification which is publicly available at https://github.com/bearpaw/pytorch-classification.
We replaced conventional convolutions with our LPSC in the models. Please also check the license and usage there if you want to make use of this code. 



## Install
* Install [PyTorch]. This code is tested in PyTorch v1.2.0.
* Please refer to https://github.com/bearpaw/pytorch-classification for other requirements. Our implementation of LPSC does not require additional dependencies or packages.



## Training 
## CIFAR-10 (We did not change any default hyper-parameters.)

#### AlexNet_LPSC
```
python cifar.py -a alexnet_lpc --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/alexnet 
```

#### VGG19(BN) LPSC
```
python cifar.py -a vgg19_bn_lpc --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn 
```

#### ResNet-20 LPSC
```
python cifar.py -a resnet_lpc --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20 
```

#### ResNet-20 LPSC-CC
```
python cifar.py -a resnet_lpsc --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20-cc 
```



## CIFAR-100 (We did not change any default hyper-parameters.)

#### AlexNet_LPSC
```
python cifar.py -a alexnet_lpc --dataset cifar100 --checkpoint checkpoints/cifar100/alexnet --epochs 164 --schedule 81 122 --gamma 0.1 
```

#### VGG19(BN) LPSC
```
python cifar.py -a vgg19_bn_lpc --dataset cifar100 --checkpoint checkpoints/cifar100/vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 
```

#### ResNet-20 LPSC
```
python cifar.py -a resnet_lpc --dataset cifar100 --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-20 
```

#### ResNet-20 LPSC-CC
```
python cifar.py -a resnet_lpsc --dataset cifar100 --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-20-cc
```



## ImageNet
#### ResNet-18 LPSC
```
python imagenet.py -a resnet18_lpc --data ./dataset/ILSVRC2012/ --train-batch 64 --lr 0.025 --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet18
```

```
python imagenet.py -a resnet18_lpsc --data ./dataset/ILSVRC2012/ --train-batch 64 --lr 0.025 --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet18
```