import numpy as np
from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, load_model
from keras import regularizers


class PIC_CLASSIFY:
    def __init__(self):
        self.MODELS = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "inception": InceptionV3,
            "xception": Xception,
            "resnet": ResNet50
        }

    def VGG19_model(self, input_shape, FREEZE_LAYER, CLASSIFY):
        model_vgg = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)

        for layer in model_vgg.layers[:-FREEZE_LAYER]:
            layer.trainable = False
        for layer in model_vgg.layers[-FREEZE_LAYER:]:
            layer.trainable = True

        model_self = Flatten(name='flatten')(model_vgg.output)
        # model_self = Dense(2048, activation='relu', name='fc1')(model_self)
        model_self = Dense(2048, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l1(0.001))(model_self)
        model_self = Dropout(0.5)(model_self)
        # model_self = Dense(1024, activation='relu', name='fc2')(model_self)
        # model_self = Dropout(0.5)(model_self)
        model_self = Dense(CLASSIFY, activation='softmax')(model_self)
        model_vgg_102 = Model(inputs=model_vgg.input, outputs=model_self, name='VGG19')
        model_vgg_102.summary()
        return model_vgg_102

    def ResNet50_model(self, input_shape, FREEZE_LAYER, CLASSIFY):
        model_vgg = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape, pooling='avg',
                             classes=CLASSIFY)
        if FREEZE_LAYER != 0:
            for layer in model_vgg.layers[:-FREEZE_LAYER]:
                layer.trainable = False
            for layer in model_vgg.layers[-FREEZE_LAYER:]:
                layer.trainable = True

        model_self = Dense(2048, activation='relu', name='fc1')(model_vgg.output)
        # model_self = Dense(1048, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01),
        #                    activity_regularizer=regularizers.l1(0.001))(model_vgg.output)
        model_self = Dropout(0.5)(model_self)
        #model_self = Dense(1048, activation='relu', name='fc2')(model_self)
        # model_self = Dense(1048, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(0.01),
        #                    activity_regularizer=regularizers.l1(0.001))(model_self)
        #model_self = Dropout(0.5)(model_self)
        # model_self = Dense(1048, activation='relu', name='fc2')(model_self)
        # model_self = Dropout(0.5)(model_self)
        model_self = Dense(CLASSIFY, activation='softmax')(model_self)
        model_vgg_102 = Model(inputs=model_vgg.input, outputs=model_self, name='ResNet50')
        model_vgg_102.summary()
        return model_vgg_102


    def InceptionV3_model(self, input_shape, FREEZE_LAYER, CLASSIFY):
        model_vgg = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape, pooling='avg',
                             classes=CLASSIFY)
        if FREEZE_LAYER != 0:
            for layer in model_vgg.layers[:-FREEZE_LAYER]:
                layer.trainable = False
            for layer in model_vgg.layers[-FREEZE_LAYER:]:
                layer.trainable = True

        model_self = Flatten(name='flatten')(model_vgg.output)
        model_self = Dense(1048, activation='relu', name='fc1')(model_self)
        # model_self = Dense(1048, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01),
        #                    activity_regularizer=regularizers.l1(0.001))(model_self)
        model_self = Dropout(0.5)(model_self)
        # model_self = Dense(1048, activation='relu', name='fc2')(model_self)
        # model_self = Dropout(0.5)(model_self)
        model_self = Dense(CLASSIFY, activation='softmax')(model_self)
        model_vgg_102 = Model(inputs=model_vgg.input, outputs=model_self, name='InceptionV3')
        model_vgg_102.summary()
        return model_vgg_102

    def Xception_model(self, input_shape, FREEZE_LAYER, CLASSIFY):
        model_vgg = Xception(include_top=False, weights="imagenet", input_shape=input_shape, pooling='avg',
                             classes=CLASSIFY)
        if FREEZE_LAYER != 0:
            for layer in model_vgg.layers[:-FREEZE_LAYER]:
                layer.trainable = False
            for layer in model_vgg.layers[-FREEZE_LAYER:]:
                layer.trainable = True

        model_self = Flatten(name='flatten')(model_vgg.output)
        model_self = Dense(1048, activation='relu', name='fc1')(model_self)
        # model_self = Dense(1048, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01),
        #                    activity_regularizer=regularizers.l1(0.001))(model_self)
        model_self = Dropout(0.5)(model_self)
        # model_self = Dense(1048, activation='relu', name='fc2')(model_self)
        # model_self = Dropout(0.5)(model_self)
        model_self = Dense(CLASSIFY, activation='softmax')(model_self)
        model_vgg_102 = Model(inputs=model_vgg.input, outputs=model_self, name='InceptionV3')
        model_vgg_102.summary()
        return model_vgg_102
