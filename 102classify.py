import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.preprocessing.image import img_to_array
from kerasmodel import img_classify_model

KEY = "vgg19"
# IMAGE_SIZE = 75
IMAGE_SIZE = 128
EPOCHS_SIZE = 30
BATCH_SIZE = 32
INIT_LR = 0.1  # 1e-3
DECAY = 1e-3
MOMENTUM = 0.9
FREEZE_LAYER = 0  # 0全部参与训练   大约0从后向前冻结   等于-1都不训练
NB_IV3_LAYERS_TO_FREEZE = 172
FC_SIZE = 2048
DROPOUT = 0.5
CLASSIFY = 102

is_load_model = False
model_save_part_path = r'data\model_' + KEY + '_model_'
model_save_path = r'data\model_' + KEY + '_model.h5'
path = r'data\train_data'
a = pd.read_csv(r'data\train.csv')
filesname = a['filename']
test_path = r'data\test_data'
b = pd.read_csv(r'data\test.csv')
t_filesname = b['filename']

labels = a['label']
encoder = LabelEncoder()
encoder.fit(labels)


# 图片读取
def load_image(path, filesname, height, width, channels):
    images = []
    for image_name in filesname:
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.resize(image, (height, width))
        images.append(img_to_array(image))
        print("已加载:"+str(len(images))+"张图片")
    images = np.array(images, dtype="float") / 255.0
    images = images.reshape([-1, height, width, channels])
    #images = np.expand_dims(images, axis=0)
    print(images.shape)
    return images


# one hot label
def label2vec(labels):
    labels1 = encoder.transform(labels)
    one_hot_labels1 = to_categorical(labels1, num_classes=102)
    return labels1, one_hot_labels1


# one hot coding 转回字符串label
def vec2label(label_vec):
    label = encoder.inverse_transform(label_vec.flatten())
    return label.reshape(-1, 5)


# 划分训练集和测试集
def make_train_and_val_set(dataset, labels, test_size):
    train_set, val_set, train_label, val_label = train_test_split(dataset, labels,
                                                                  test_size=test_size, random_state=5)
    return train_set, val_set, train_label, val_label

# 取预测值的前k位
def get_top_k_label(preds, k=1):
    top_k = tf.nn.top_k(preds, k).indices
    with tf.Session() as sess:
        top_k = sess.run(top_k)
    top_k_label = vec2label(top_k)
    return top_k_label


# 数据增强
# train_datagen = ImageDataGenerator(width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    fill_mode='nearest')
train_datagen = ImageDataGenerator(
    #随机转动的角度
    #rotation_range=30,
    # 随机水平偏移的幅度
    width_shift_range=0.2,
    # 随机竖直偏移的幅度
    height_shift_range=0.2,
    # 剪切变换的程度
    shear_range=0.2,
    # 随机的放大
    zoom_range=0.2,
    # 随机水平翻转
    horizontal_flip=True,
    # 布尔值，进行随机竖直翻转
    # vertical_flip=True,
    # featurewise_center=False,
    # zca_whitening=True,
    # ‘constant’，‘nearest’，‘reflect’或‘wrap’之一
    # 当进行变换时超出边界的点将根据本参数给定的方法进行处理
    # fill_mode='nearest'
)

tb = TensorBoard(log_dir='data/TensorBoard/logs_self',  # log 目录
                 histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=False, # 是否可视化梯度直方图
                 write_images=False,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)


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

    def loss_plot(self, loss_type, loss_img_path="data/loss.png"):
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
        plt.savefig(loss_img_path)
        plt.show()


if __name__ == '__main__':
    print("划分训练集和测试集")
    train_set_name, val_set_name, train_label, val_label = make_train_and_val_set(filesname, labels, 0.2)
    print("加载训练集图片")
    train_set = load_image(path, train_set_name, IMAGE_SIZE, IMAGE_SIZE, 3)
    train_label0, train_label1 = label2vec(train_label)
    print("加载验证集图片")
    val_set = load_image(path, val_set_name, IMAGE_SIZE, IMAGE_SIZE, 3)
    val_label0, val_label1 = label2vec(val_label)
    print("准备损失函数图像")
    history = LossHistory()
    # windows上执行以下命令日志路径要用双引号，否则读取不到
    # tensorboard --logdir="E:\code\Python3\machine_learnin\kerasmodel\data\TensorBoard\logs"
    earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    callback = [history, tb, TensorBoard(log_dir='data/TensorBoard/logs')]
    print("创建模型")
    classify_model = img_classify_model.PIC_CLASSIFY([IMAGE_SIZE, IMAGE_SIZE, 3], FREEZE_LAYER, CLASSIFY, FC_SIZE, DROPOUT)
    model = classify_model.getInstance(KEY)

    # 载入模型
    if is_load_model:
        print("设定载入模型！！！")
        if not os.path.exists(model_save_path):
            print("模型不存在！！！")
    if is_load_model and os.path.exists(model_save_path):
        print("载入已有模型：" + model_save_path)
        model = load_model(model_save_path)

    #opt = SGD(lr=INIT_LR, momentum=MOMENTUM)
    #opt = SGD(lr=INIT_LR, momentum=MOMENTUM, decay=DECAY)
    opt = Adam() #lr=INIT_LR, decay=INIT_LR / EPOCHS_SIZE
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=SGD(),
                            metrics=['accuracy'])

    #model.summary()
    for i in range(round(EPOCHS_SIZE / 5)):
        print("EPOCHS_SIZE：" + str(EPOCHS_SIZE))
        print("第" + str(i + 1) + "个5轮epochs开始")
        print("训练开始")
        steps_per_epoch = (len(train_set) + BATCH_SIZE - 1) // BATCH_SIZE
        model.fit_generator(train_datagen.flow(train_set, train_label1, batch_size=BATCH_SIZE),
                            steps_per_epoch=steps_per_epoch, epochs=5,
                            validation_data=(val_set, val_label1), callbacks=callback)

        print("训练结束")
        print("绘制第" + str(i + 1) + "张损失函数图像")
        path = "data/loss" + str(i + 1) + ".png"
        history.loss_plot('epoch', path)

        # 保存模型
        print("保存" + str(i+1) + "模型开始")
        model.save(model_save_part_path + str(str(i + 1)) + ".h5")
        print("保存" + str(i+1) + "模型结束")
        print("第" + str(i+1) + "个5轮epochs结束")

    print("保存模型开始")
    model.save(model_save_path)
    print("保存模型结束")

    print("加载测试图片")
    test_set = load_image(test_path, t_filesname, IMAGE_SIZE, IMAGE_SIZE, 3)
    print("预测测试集开始")
    test_preds = model.predict(test_set)
    print("预测测试集结束")
    print(test_preds)

    predslabel = get_top_k_label(test_preds, 5)

    submit = pd.DataFrame({'fliesname': t_filesname, 'pred1': predslabel[:, 0],
                           'pred2': predslabel[:, 1], 'pred3': predslabel[:, 2],
                           'pred4': predslabel[:, 3], 'pred5': predslabel[:, 4]})
    print("保存预测结果")
    submit.to_csv(r'data\submit.csv', index=False)
    print("保存完毕")
    print("end")
