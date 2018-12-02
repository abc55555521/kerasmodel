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

import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import plot_model, to_categorical
from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
from keras.applications import imagenet_utils
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
import cv2


class LossHistory(Callback):
    def __init__(self):
        self.losses = {}
        self.accuracy = {}
        self.val_loss = {}
        self.val_acc = {}

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        # 在 plt.show() 后调用了 plt.savefig() ，在 plt.show() 后实际上已经创建了一个新的空白的图片（坐标轴），这时候你再 plt.savefig() 就会保存这个新生成的空白图片
        # plt.savefig 一定要在plt.show之前调用
        plt.savefig("data/acc_loss.png")
        plt.show()


class pic_classify:
    def __init__(self):
        self.IMAGE_SIZE = 64
        self.EPOCHS_SIZE = 2
        self.BATCH_SIZE = 32
        self.CLASSIFY = 102
        self.INIT_LR = 1e-3
        self.DECAY = 1e-5
        self.MOMENTUM = 0.9
        self.FREEZE_LAYER = 5
        self.MODELS = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "inception": InceptionV3,
            "xception": Xception,
            "resnet": ResNet50
        }
        self.history = LossHistory()

        # 数据增强
        self.train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.tb = TensorBoard(log_dir='data/TensorBoard/logs_self',  # log 目录
                              histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                              batch_size=32,  # 用多大量的数据计算直方图
                              write_graph=True,  # 是否存储网络结构图
                              write_grads=False,  # 是否可视化梯度直方图
                              write_images=False,  # 是否可视化参数
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)
        self.ImageShape_224 = (224, 224, 3)
        self.ImageShape_299 = (299, 299, 3)

        self.path = r'data\train_data'
        a = pd.read_csv(r'data\train.csv')
        self.filesname = a['filename']
        self.test_path = r'data\test_data'
        b = pd.read_csv(r'data\test.csv')
        self.t_filesname = b['filename']

        self.labels = a['label']
        self.encoder = LabelEncoder()
        self.encoder.fit(self.labels)

    # 图片读取
    def load_image(self, path, filesname, height, width, channels):
        images = []
        for image_name in filesname:
            image = cv2.imread(os.path.join(path, image_name))
            image = cv2.resize(image, (height, width))
            images.append(img_to_array(image))
            print("已加载:"+str(len(images))+"张图片")
        images = np.array(images, dtype="float") / 255.0
        images = images.reshape([-1, height, width, channels])
        print(images.shape)
        return images

    # one hot label
    def label2vec(self, labels):
        labels1 = self.encoder.transform(labels)
        one_hot_labels1 = to_categorical(labels1, num_classes=102)
        return labels1, one_hot_labels1

    # one hot coding 转回字符串label
    def vec2label(self, label_vec):
        label = self.encoder.inverse_transform(label_vec)
        return label


    # 划分训练集和测试集
    def make_train_and_val_set(self, dataset, labels, test_size):
        train_set, val_set, train_label, val_label = train_test_split(dataset, labels,
                                                                      test_size=test_size, random_state=5)
        return train_set, val_set, train_label, val_label

    # 取预测值的前k位
    def get_top_k_label(self, preds, k=1):
        top_k = tf.nn.top_k(preds, k).indices
        with tf.Session() as sess:
            top_k = sess.run(top_k)
        top_k_label = self.vec2label(top_k)
        return top_k_label

    # VGG模型
    def VGG19_model(self, input_shape,
                    is_plot_model=False):
        base_model = VGG19(weights='imagenet', include_top=False, pooling=None,
                           input_shape=input_shape,
                           classes=self.CLASSIFY)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        layer_index = 1
        for layer in base_model.layers:
            if layer_index < self.FREEZE_LAYER + 1:
                layer.trainable = False
            layer_index += 1

        x = base_model.output
        # 添加自己的全链接分类层
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.CLASSIFY, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=self.INIT_LR, decay=self.DECAY, momentum=self.MOMENTUM, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘图
        if is_plot_model:
            plot_model(model, to_file='data/submit/vgg19_model.png', show_shapes=True)

        return model

    # InceptionV3模型
    def InceptionV3_model(self, input_shape, is_plot_model=False):

        base_model = InceptionV3(weights='imagenet', include_top=False, pooling=None,
                                 input_shape=input_shape,
                                 classes=self.CLASSIFY)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        # layer_index = 1
        # for layer in base_model.layers:
        #     if layer_index < self.FREEZE_LAYER + 1:
        #         layer.trainable = False
        #     layer_index += 1

        x = base_model.output
        # 添加自己的全链接分类层
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.CLASSIFY, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=self.INIT_LR, decay=self.DECAY, momentum=self.MOMENTUM, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘图
        if is_plot_model:
            plot_model(model, to_file='data/submit/inception_v3_model.png', show_shapes=True)

        return model



    # ResNet模型
    def ResNet50_model(self, input_shape, is_plot_model=True):
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape,
                              classes=self.CLASSIFY)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        # layer_index = 1
        # for layer in base_model.layers:
        #     if layer_index < self.FREEZE_LAYER + 1:
        #         layer.trainable = False
        #     layer_index += 1

        x = base_model.output
        # 添加自己的全链接分类层
        model_self = Flatten()(x)
        #x = GlobalAveragePooling2D()(x)
        model_self = Dense(1024, activation='relu')(model_self)
        predictions = Dense(self.CLASSIFY, activation='softmax')(model_self)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=self.INIT_LR, decay=self.DECAY, momentum=self.MOMENTUM, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘制模型
        if is_plot_model:
            plot_model(model, to_file='data/submit/resnet50_model.png', show_shapes=True)

        return model

    # 训练模型
    def train_model(self, model, epochs, train_generator, steps_per_epoch, validation_data,
                    model_save_url, img_save_path, callbacks, is_load_model=False):
        # 载入模型
        if is_load_model and os.path.exists(model_save_url):
            model = load_model(model_save_url)

        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks)
        # 模型保存
        print("保存模型开始")
        model.save(model_save_url, overwrite=True)
        print("保存模型结束")

        print("开始绘制训练的acc_loss图")
        transfer.plot_training(history_ft, img_save_path)
        transfer.history.loss_plot('epoch')
        print("结束绘制训练的acc_loss图")
        return model


    def plot_training(self, history, save_path):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'b-')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training and validation accuracy')
        plt.figure()
        plt.plot(epochs, loss, 'b-')
        plt.plot(epochs, val_loss, 'r-')
        plt.title('Training and validation loss')
        plt.savefig(save_path)
        plt.show()


if __name__ == '__main__':
    # for key in MODELS.keys():
    #     inputShape = ImageShape_224
    #     if key in ("inception", "xception"):
    #         inputShape = ImageShape_299
    #     Network = MODELS(key)
    #     model = Network(weights="imagenet", include_top=False, input_shape=inputShape)
    transfer = pic_classify()
    # 得到数据
    print("划分训练集和测试集")
    train_set_name, val_set_name, train_label, val_label = transfer.make_train_and_val_set(transfer.filesname,
                                                                                           transfer.labels, 0.2)
    print("加载训练集图片")
    train_set = transfer.load_image(transfer.path, train_set_name, transfer.IMAGE_SIZE, transfer.IMAGE_SIZE, 3)
    train_label0, train_label1 = transfer.label2vec(train_label)
    print("加载验证集图片")
    val_set = transfer.load_image(transfer.path, val_set_name, transfer.IMAGE_SIZE, transfer.IMAGE_SIZE, 3)
    val_label0, val_label1 = transfer.label2vec(val_label)
    print("准备损失函数图像")

    callbacks = [transfer.history, transfer.tb, TensorBoard(log_dir='data/TensorBoard/logs')]

    generator = transfer.train_datagen.flow(train_set, train_label1, batch_size=transfer.BATCH_SIZE)

    # VGG19
    # print("创建VGG19模型")
    # model_save_path = 'vgg19_model.h5'
    # img_save_path = "data/submit/vgg19_acc_loss.png"
    # predict_path = "data/submit/vgg19_submit.csv"
    # model = transfer.VGG19_model(input_shape=(transfer.IMAGE_SIZE, transfer.IMAGE_SIZE, 3), is_plot_model=True)
    # print("训练开始")
    # model = transfer.train_model(model, transfer.EPOCHS_SIZE, generator,
    #                              steps_per_epoch=len(train_set) // transfer.BATCH_SIZE,
    #                              validation_data=(val_set, val_label1), callbacks=callbacks,
    #                              model_save_url=model_save_path, img_save_path=img_save_path, is_load_model=False)
    # print("训练结束")

    # ResNet50
    print("创建ResNet50模型")
    model_save_path = 'data/submit/resnet50_model.h5'
    img_save_path = "data/submit/resnet50_acc_loss.png"
    predict_path = "data/submit/resnet50_submit.csv"
    model = transfer.ResNet50_model(input_shape=(transfer.IMAGE_SIZE, transfer.IMAGE_SIZE, 3), is_plot_model=True)
    print("训练ResNet50模型开始")
    model = transfer.train_model(model, transfer.EPOCHS_SIZE, generator,
                                 steps_per_epoch=len(train_set) // transfer.BATCH_SIZE,
                                 validation_data=(val_set, val_label1), callbacks=callbacks,
                                 model_save_url=model_save_path, img_save_path=img_save_path, is_load_model=False)
    print("训练ResNet50模型结束")

    # InceptionV3
    # print("创建InceptionV3模型")
    # model_save_path = 'data/submit/inceptionV3_model.h5'
    # img_save_path = "data/submit/inceptionV3_acc_loss.png"
    # predict_path = "data/submit/inceptionV3_submit.csv"
    # model = transfer.InceptionV3_model(input_shape=(transfer.IMAGE_SIZE, transfer.IMAGE_SIZE, 3), is_plot_model=True)
    # print("训练InceptionV3模型开始")
    # model = transfer.train_model(model, transfer.EPOCHS_SIZE, generator,
    #                              steps_per_epoch=len(train_set) // transfer.BATCH_SIZE,
    #                              validation_data=(val_set, val_label1), callbacks=callbacks,
    #                              model_save_url=model_save_path, img_save_path=img_save_path, is_load_model=False)
    # print("训练InceptionV3模型结束")


    print("加载测试集")
    test_set = transfer.load_image(transfer.test_path, transfer.t_filesname, transfer.IMAGE_SIZE, transfer.IMAGE_SIZE, 3)
    print("预测测试集开始")
    test_preds = model.predict(test_set)
    print("预测测试集结束")
    print(test_preds)
    print("获取top5")
    predslabel = transfer.get_top_k_label(test_preds, 5)
    submit = pd.DataFrame({'fliesname': transfer.t_filesname, 'pred1': predslabel[:, 0],
                           'pred2': predslabel[:, 1], 'pred3': predslabel[:, 2],
                           'pred4': predslabel[:, 3], 'pred5': predslabel[:, 4]})
    print("保存预测结果")
    submit.to_csv(predict_path, index=False)
    print("保存完毕")
    print("end")

