# 五个卷积神经网络模型
# VGG16
# VGG19
# ResNet50
# Inception V3
# Xception

# ###################### VGG16 与 VGG19#########################################################################
# VGG有两个很大的缺点：
# 1.网络架构weight数量相当大，很消耗磁盘空间。
# 2.训练非常慢
# 由于其全连接节点的数量较多，再加上网络比较深，VGG16有533MB+，VGG19有574MB。这使得部署VGG比较耗时。
# 我们仍然在很多深度学习的图像分类问题中使用VGG，然而，较小的网络架构通常更为理想（例如SqueezeNet、GoogLeNet等）
# ###################### ResNet（残差网络）#####################################################################
# ###################### Inception V3 ##########################################################################
# ###################### Xception ##############################################################################

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2


IMAGE_SIZE = 64
EPOCHS_SIZE = 2
BATCH_SIZE = 32
INIT_LR = 1e-3
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

ImageShape_224 = (224, 224, 3)
ImageShape_299 = (299, 299, 3)
preprocess = imagenet_utils.preprocess_input

if __name__ == '__main__':
    for key in MODELS.keys():
        inputShape = ImageShape_224
        if key in ("inception", "xception"):
            inputShape = ImageShape_299
        Network = MODELS(key)
        model = Network(weights="imagenet", include_top=False, input_shape=inputShape)
