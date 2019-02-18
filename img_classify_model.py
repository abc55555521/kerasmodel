import numpy as np
from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import regularizers


class PIC_CLASSIFY:
    def __init__(self, input_shape, FREEZE_LAYER, CLASSIFY, FC_SIZE, DROPOUT):

        self.INPUT_SHAPE = input_shape
        self.FREEZE_LAYER = FREEZE_LAYER
        self.CLASSIFY = CLASSIFY
        self.FC_SIZE = FC_SIZE
        self.DROPOUT = DROPOUT

    def getInstance(self, key):
        MODELS = {
            "vgg16": self.VGG19_model,
            "vgg19": self.VGG19_model,
            "inception": self.InceptionV3_model,
            "xception": self.Xception_model,
            "resnet": self.ResNet50_model
        }
        keys = ["vgg16", "vgg19", "inception", "xception", "resnet"]
        print("模型key:" +key)
        return MODELS.get(key)()

    def VGG19_model(self):
        print("创建VGG19模型")
        model_vgg = VGG19(include_top=False, weights="imagenet", input_shape=self.INPUT_SHAPE)

        if self.FREEZE_LAYER > 0:
            for layer in model_vgg.layers[:-self.FREEZE_LAYER]:
                layer.trainable = False
            for layer in model_vgg.layers[-self.FREEZE_LAYER:]:
                layer.trainable = True
        elif self.FREEZE_LAYER == -1:
            for layer in model_vgg.layers:
                layer.trainable = False
        else:
            for layer in model_vgg.layers:
                layer.trainable = True

        # model_self = GlobalAveragePooling2D()(model_vgg.output)
        model_self = Flatten(name='flatten')(model_vgg.output)
        model_self = Dense(self.FC_SIZE, activation='relu', name='fc1')(model_self)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01),
        #                    activity_regularizer=regularizers.l1(0.001))(model_self)
        model_self = Dropout(self.DROPOUT)(model_self)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc2')(model_self)
        # model_self = Dropout(self.DROPOUT)(model_self)
        model_self = Dense(self.CLASSIFY, activation='softmax')(model_self)
        model_vgg_102 = Model(inputs=model_vgg.input, outputs=model_self, name='VGG19')
        model_vgg_102.summary()
        return model_vgg_102

    def ResNet50_model(self):
        print("创建ResNet50模型")
        model_vgg = ResNet50(include_top=False, weights="imagenet", input_shape=self.INPUT_SHAPE, pooling='avg',
                             classes=self.CLASSIFY)
        if self.FREEZE_LAYER != 0:
            for layer in model_vgg.layers[:-self.FREEZE_LAYER]:
                layer.trainable = False
            for layer in model_vgg.layers[-self.FREEZE_LAYER:]:
                layer.trainable = True
        else:
            for layer in model_vgg.layers:
                layer.trainable = False

        model_self = Dense(self.FC_SIZE, activation='relu', name='fc1')(model_vgg.output)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01),
        #                    activity_regularizer=regularizers.l1(0.001))(model_vgg.output)
        model_self = Dropout(self.DROPOUT)(model_self)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc2')(model_self)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(0.01),
        #                    activity_regularizer=regularizers.l1(0.001))(model_self)
        # model_self = Dropout(self.DROPOUT)(model_self)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc2')(model_self)
        # model_self = Dropout(self.DROPOUT)(model_self)
        model_self = Dense(self.CLASSIFY, activation='softmax')(model_self)
        model_vgg_102 = Model(inputs=model_vgg.input, outputs=model_self, name='ResNet50')
        model_vgg_102.summary()
        return model_vgg_102


    def InceptionV3_model(self):
        print("创建InceptionV3模型")
        model_vgg = InceptionV3(include_top=False, weights="imagenet", input_shape=self.INPUT_SHAPE, pooling='avg',
                             classes=self.CLASSIFY)
        if self.FREEZE_LAYER != 0:
            for layer in model_vgg.layers[:-self.FREEZE_LAYER]:
                layer.trainable = False
            for layer in model_vgg.layers[-self.FREEZE_LAYER:]:
                layer.trainable = True

        #model_self = Flatten(name='flatten')(model_vgg.output)
        model_self = Dense(self.FC_SIZE, activation='relu', name='fc1')(model_vgg.output)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01),
        #                    activity_regularizer=regularizers.l1(0.001))(model_self)
        model_self = Dropout(self.DROPOUT)(model_self)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc2')(model_self)
        # model_self = Dropout(self.DROPOUT)(model_self)
        model_self = Dense(self.CLASSIFY, activation='softmax')(model_self)
        model_vgg_102 = Model(inputs=model_vgg.input, outputs=model_self, name='InceptionV3')
        model_vgg_102.summary()
        return model_vgg_102

    def Xception_model(self):
        print("创建Xception模型")
        model_vgg = Xception(include_top=False, weights="imagenet", input_shape=self.INPUT_SHAPE, pooling='avg',
                             classes=self.CLASSIFY)
        if self.FREEZE_LAYER != 0:
            for layer in model_vgg.layers[:-self.FREEZE_LAYER]:
                layer.trainable = False
            for layer in model_vgg.layers[-self.FREEZE_LAYER:]:
                layer.trainable = True

        # model_self = Flatten(name='flatten')(model_vgg.output)
        model_self = Dense(self.FC_SIZE, activation='relu', name='fc1')(model_vgg.output)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01),
        #                    activity_regularizer=regularizers.l1(0.001))(model_self)
        model_self = Dropout(self.DROPOUT)(model_self)
        # model_self = Dense(self.FC_SIZE, activation='relu', name='fc2')(model_self)
        # model_self = Dropout(self.DROPOUT)(model_self)
        model_self = Dense(self.CLASSIFY, activation='softmax')(model_self)
        model_vgg_102 = Model(inputs=model_vgg.input, outputs=model_self, name='Xception')
        model_vgg_102.summary()
        return model_vgg_102
