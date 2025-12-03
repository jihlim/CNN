import os

from models.alexnet import *
from models.convnext import *
from models.efficientnet import *
from models.efficientnetv2 import *
from models.mnasnet import *
from models.mobilenet import *
from models.mobilenetv2 import * 
from models.mobilenetv3 import * 
from models.nfnet import *
from models.regnet import *
from models.resnet import *
from models.resnext import *
from models.senet import *
from models.shufflenet import *
from models.shufflenetv2 import *
from models.xception import *
from models.vgg import *

def get_model(model_name, num_classes, device):

    # AlexNet
    if model_name.lower() == "alexnet":
        model = alexnet(num_classes)
    
    # ConvNeXt Series
    #   - ConvNeXt V1
    elif model_name.lower() == "convnext_t":
        model = convnext_t(num_classes)
    
    elif model_name.lower() == "convnext_s":
        model = convnext_s(num_classes)
    
    elif model_name.lower() == "convnext_b":
        model = convnext_b(num_classes)
    
    elif model_name.lower() == "convnext_l":
        model = convnext_l(num_classes)
    
    elif model_name.lower() == "convnext_xl":
        model = convnext_xl(num_classes)

    # EfficientNet Series
    #   - EfficientNet V1
    elif model_name.lower() == "efficientnet_b0":
        model = efficientnet_b0(num_classes)

    elif model_name.lower() == "efficientnet_b1":
        model = efficientnet_b1(num_classes)

    elif model_name.lower() == "efficientnet_b2":
        model = efficientnet_b2(num_classes)
    
    elif model_name.lower() == "efficientnet_b3":
        model = efficientnet_b3(num_classes)
    
    elif model_name.lower() == "efficientnet_b4":
        model = efficientnet_b4(num_classes)

    elif model_name.lower() == "efficientnet_b5":
        model = efficientnet_b5(num_classes)
    
    elif model_name.lower() == "efficientnet_b6":
        model = efficientnet_b6(num_classes)

    elif model_name.lower() == "efficientnet_b7":
        model = efficientnet_b7(num_classes)
    
    #   - EfficientNet V2
    elif model_name.lower() == "efficientnetv2_b":
        model = efficientnetv2_base(num_classes)

    elif model_name.lower() == "efficientnetv2_s":
        model = efficientnetv2_S(num_classes)

    elif model_name.lower() == "efficientnetv2_m":
        model = efficientnetv2_M(num_classes)

    elif model_name.lower() == "efficientnetv2_l":
        model = efficientnetv2_L(num_classes)
    
    elif model_name.lower() == "efficientnetv2_xl":
        model = efficientnetv2_XL(num_classes)
    
    # MnasNet
    elif model_name.lower() == "mnasnet_a1":
        model = mnasnet_a1(num_classes)

    # MobileNet Series
    #   - MobileNet V1
    elif model_name.lower() == "mobilenet":
        model = mobilenet(num_classes)
    
    #   - MobileNet V2
    elif model_name.lower() == "mobilenetv2":
        model = mobilenetv2(num_classes)
    
    #   - MobileNet V3
    elif model_name.lower() == "mobilenetv3_small":
        model = mobilenetv3_small(num_classes)
    
    elif model_name.lower() == "mobilenetv3_large":
        model = mobilenetv3_large(num_classes)

    # NFNet
    elif model_name.lower() == "nfnet_f0":
        model = nfnet_f0(num_classes)
    
    elif model_name.lower() == "nfnet_f1":
        model = nfnet_f1(num_classes)
    
    elif model_name.lower() == "nfnet_f2":
        model = nfnet_f2(num_classes)
    
    elif model_name.lower() == "nfnet_f3":
        model = nfnet_f3(num_classes)
    
    elif model_name.lower() == "nfnet_f4":
        model = nfnet_f4(num_classes)
    
    elif model_name.lower() == "nfnet_f5":
        model = nfnet_f5(num_classes)
    
    elif model_name.lower() == "nfnet_f6":
        model = nfnet_f6(num_classes)

    # RegNet
    # RegNetX
    elif model_name.lower() == "regnetx_200m":
        model = regnetx_200M(num_classes)

    elif model_name.lower() == "regnetx_400m":
        model = regnetx_400M(num_classes)
    
    elif model_name.lower() == "regnetx_600m":
        model = regnetx_600M(num_classes)
    
    elif model_name.lower() == "regnetx_800m":
        model = regnetx_800M(num_classes)

    elif model_name.lower() == "regnetx_1.6g":
        model = regnetx_1_6G(num_classes)

    elif model_name.lower() == "regnetx_3.2g":
        model = regnetx_3_2G(num_classes)
    
    elif model_name.lower() == "regnetx_4g":
        model = regnetx_4G(num_classes)
    
    elif model_name.lower() == "regnetx_6.4g":
        model = regnetx_6_4G(num_classes)

    elif model_name.lower() == "regnetx_8g":
        model = regnetx_8G(num_classes)
    
    elif model_name.lower() == "regnetx_12g":
        model = regnetx_12G(num_classes)

    elif model_name.lower() == "regnetx_16g":
        model = regnetx_16G(num_classes)

    elif model_name.lower() == "regnetx_32g":
        model = regnetx_32G(num_classes)
    
    # RegNetY
    elif model_name.lower() == "regnety_200m":
        model = regnety_200M(num_classes)

    elif model_name.lower() == "regnety_400m":
        model = regnety_400M(num_classes)
    
    elif model_name.lower() == "regnety_600m":
        model = regnety_600M(num_classes)
    
    elif model_name.lower() == "regnety_800m":
        model = regnety_800M(num_classes)

    elif model_name.lower() == "regnety_1.6g":
        model = regnety_1_6G(num_classes)

    elif model_name.lower() == "regnety_3.2g":
        model = regnety_3_2G(num_classes)
    
    elif model_name.lower() == "regnety_4g":
        model = regnety_4G(num_classes)
    
    elif model_name.lower() == "regnety_6.4g":
        model = regnety_6_4G(num_classes)

    elif model_name.lower() == "regnety_8g":
        model = regnety_8G(num_classes)
    
    elif model_name.lower() == "regnety_12g":
        model = regnety_12G(num_classes)

    elif model_name.lower() == "regnety_16g":
        model = regnety_16G(num_classes)

    elif model_name.lower() == "regnety_32g":
        model = regnety_32G(num_classes)

    # ResNet 
    elif model_name.lower() == "resnet18":
        model = resnet18(num_classes)
    
    elif model_name.lower() == "resnet20":
        model = resnet20(num_classes)
    
    elif model_name.lower() == "resnet20_greyscale":
        model = resnet20_greyscale(num_classes)
    
    elif model_name.lower() == "resnet32":
        model = resnet32(num_classes)
    
    elif model_name.lower() == "resnet34":
        model = resnet34(num_classes)
    
    elif model_name.lower() == "resnet50":
        model = resnet50(num_classes)
    
    elif model_name.lower() == "resnet101":
        model = resnet101(num_classes)
    
    elif model_name.lower() == "resnet152":
        model = resnet152(num_classes)
    
    # ResNeXt
    elif model_name.lower() == "resnext26":
        model = resnext26(num_classes)
    
    elif model_name.lower() == "resnext50":
        model = resnext50(num_classes)
    
    # SENet
    elif model_name.lower() == "senet18":
        model = senet18(num_classes)

    elif model_name.lower() == "senet50":
        model = senet50(num_classes)
    
    # ShuffleNet Series
    # ShuffleNet
    elif model_name.lower() == "shufflenet_g1":
        model = shufflenet_g1(num_classes)
    
    elif model_name.lower() == "shufflenet_g2":
        model = shufflenet_g2(num_classes)
    
    elif model_name.lower() == "shufflenet_g3":
        model = shufflenet_g3(num_classes)
    
    elif model_name.lower() == "shufflenet_g4":
        model = shufflenet_g4(num_classes)
    
    elif model_name.lower() == "shufflenet_g8":
        model = shufflenet_g8(num_classes)

    # ShuffleNet V2
    elif model_name.lower() == "shufflenetv2_50":
        model = shufflenetv2_50(num_classes)

    # VGG
    elif model_name.lower() == "vgg11":
        model = vgg11(num_classes)
    
    elif model_name.lower() == "vgg11_lrn":
        model = vgg11_lrn(num_classes)
    
    elif model_name.lower() == "vgg13":
        model = vgg13(num_classes)
    
    elif model_name.lower() == "vgg16_c":
        model = vgg16_c(num_classes)
    
    elif model_name.lower() == "vgg16":
        model = vgg16(num_classes)
    
    elif model_name.lower() == "vgg19":
        model = vgg19(num_classes)

    # Xception
    elif model_name.lower() == "xception71":
        model = xception71(num_classes)

    return model.to(device)

def load_model(cfg, model, model_name, output_path):
    model_type = cfg["model"]["pretrained_type"]
    checkpoint_path = os.path.join(output_path, f"{model_name}_{model_type}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = int(checkpoint['epoch']) + 1
        print("[INFO] \t Load the Checkpoint...")
    else:
        raise Exception("[INFO] \t PRETRAINED MODEL DOES NOT EXIST! Please Train a Model from Scratch!")
    
    return model, start_epoch