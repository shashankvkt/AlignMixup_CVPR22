# AlignMixup (CVPR 2022)
This repo consists of the official Pytorch code for our CVPR 2022 paper AlignMixup: Improving Representations By Interpolating Aligned Features (http://arxiv.org/abs/2103.15375) 

### Requirements
This code has been tested with  
python 3.8.11  
torch 1.10.1  
torchvision 0.11.2  
numpy==1.21.0

### Additional package versions
cuda 11.3.1  
cudnn 8.2.0.53-11.3   
tar==1.34  
py-virtualenv==16.7.6


### Dataset Preparation  

1. For CIFAR-10/100, the dataset will automatically be downloaded, if there does not exist any CIFAR-10/100 directory in the path specified while executing the code.  
2. For Tiny-Imagenet-200, you can download the dataset from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip). Unzip it and specify its path in the code.  

Alternatively, you can run the following command in your terminal if you have ```wget``` installed to download it to your current directory:  
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```


### How to run experiments for CIFAR-10

#### AlignMixup PreActResnet18
```
cd cifar10_100  

python main.py --dataset cifar10 --data_dir path_to_cifar10_directory \
		--save_dir path_to_save_checkpoints --network resnet --epochs 2000 \
		--alpha 2.0 --num_classes 10 --manualSeed 8492 
```

####  AlignMixup WRN 16x8
```
cd cifar10_100  

python main.py --dataset cifar10 --data_dir path_to_cifar10_directory \
		--save_dir path_to_save_checkpoints --network wideresnet --epochs 2000 \
		--alpha 2.0 --num_classes 10 --manualSeed 8492
```



### How to run experiments for CIFAR-100

#### AlignMixup PreActResnet18
```
cd cifar10_100  

python main.py --dataset cifar100 --data_dir path_to_cifar100_directory \
		--save_dir path_to_save_checkpoints --network resnet --epochs 2000 \
		--alpha 2.0 --num_classes 100 --manualSeed 8492 
```

####  AlignMixup WRN 16x8
```
cd cifar10_100  

python main.py --dataset cifar100 --data_dir path_to_cifar100_directory \
		--save_dir path_to_save_checkpoints --network wideresnet --epochs 2000 \
		--alpha 2.0 --num_classes 100 --manualSeed 8492 
```


### How to run experiments for Tiny-Imagenet-200

  
#### AlignMixup PreActResnet18
```
cd tiny_imgnet  

python main.py  --train_dir path_to_train_directory \
		--val_dir path_to_val_directory \
		--save_dir path_to_save_checkpoints --epochs 1200 \
		--alpha 2.0 --num_classes 200 --manualSeed 8492
```


### How to run experiments for Imagenet

#### To run on a subset of training set (i.e approx 20% images per class)

```
cd imagenet  

python main.py --data_dir path_to_imagenet_directory --save_dir path_to_save_checkpoints \
		--mini_imagenet True --subset 260 --num_classes 1000 --epochs 300 --alpha 2.0 --batch_size 1024
```


#### To run on a full imagenet

```
cd imagenet  

python main.py --data_dir path_to_imagenet_directory --save_dir path_to_save_checkpoints \
		--mini_imagenet False --num_classes 1000 --epochs 300 --alpha 2.0 --batch_size 1024
```

#### TODO  
Imagenet using Distributed data parallel (multiple nodes) - coming soon


## Results

|  Dataset       | Network   | AlignMixup |   |
|:--------------:|:---------:|:----------:|---|
| CIFAR-10       | Resnet-18 | 97.05%     | [log](logfiles/cifar10/log_r18.txt)  |
| CIFAR-10       | WRN 16x8  | 96.91%     | [log](logfiles/cifar10/log_wrn16x8.txt) |  
| CIFAR-100      | Resnet-18 | 81.71%     | [log](logfiles/cifar100/log_r18.txt)        |  
| CIFAR-100      | WRN 16x8  | 81.23%     | [log](logfiles/cifar100/log_wrn16x8.txt)|  
| Tiny-Imagenet  | Resnet-18 | 66.87%     | [log](logfiles/tiny_imagenet/log.txt)              |  
| Imagenet       | Resnet-50 | 79.32%     | [log](logfiles/imnet/log.txt)           |  

## Acknowledgement
The code for Sinkhorn-Knopp algorithm is adapted and modified based on this amazing repository by [Daniel Daza](https://github.com/dfdazac/wassdistance)




## Citation

If you find this work useful and use it on your own research, please cite our paper  

```
@inproceedings{venkataramanan2021alignmix,
  title={AlignMixup: Improving Representations By Interpolating Aligned Features},
  author={Venkataramanan, Shashanka and Kijak, Ewa and Amsaleg, Laurent and Avrithis, Yannis},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

```
