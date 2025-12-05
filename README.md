# CNN
The unofficial PyTorch implementation of Covolutional Neural Networks (CNN)

## Updates


## Installation
```
$ git clone https://github.com/jihlim/CNN.git
$ cd CNN
$ pip install -r requirements.txt
```

## Models
- AlexNet 
- ConvNeXt
- EfficientNet
- EfficientNet V2
- MnasNet
- MobileNet
- MobileNet V2
- MobileNet V3
- NFNet
- RegNet
- ResNet
- ResNeXt
- SENet
- ShuffleNet
- ShuffleNet V2
- VGG
- Xception

## Dataset
### ImageNet-1K
- Add `ImageNet-1k` dataset and `ILSVRC2012_devkit_t12.tar` file under `data` folder
```
CNN
├── data
│   └── imagenet1k
│       ├── train
│       ├── val
│       └── ILSVRC2012_devkit_t12.tar

```

## Train
```
$ cd CNN
$ python src/train.py --model <model_name> --dataset <dataset_name>
```

Flags:
- `--model` Set model
- `--dataset` Set dataset 
- `--image_size` Set input image size
- `--epochs` Set training epochs
- `--batch_size` Set batch size
- `--lr` Set learning rate
- `--lr_decay` Set learning rate decay
- `--momentum` Set momentum
- `--wd` Set weight decay 
- `--betas` Set betas for optimizer
- `--gamma` Set gamma for scheduler
- `--milestones`Set milestones for scheduler
- `--optimizer` Set optimizer
- `--scheduler` Set Scheduler
- `--resume` Continue training

## Evaluation
```
$ cd CNN
$ python src/test.py --model <model_name> --dataset <dataset_name>
```

Flags:
- `--model` Set model 
- `--dataset` Set dataset 
- `--image_size` Set input image size
- `--batch_size` Set batch size


## Wandb (Weights & Biases)
- Type `Wandb API key` in `wandb.login()` in `src/uitls/train_utils.py`