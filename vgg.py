import cv2
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, load_model
import tensorflow as tf
import random
# 训练集路径
path = r'data\train_data'
# 测试集路径
test_path = r'data\test_data'
# 训练集excel路径
file = pd.read_csv(r'data\train.csv')
# 导出文件路径
export_file = r'data\test.csv'
filesname = file['filename']
labels = file['label']
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
EPOCHS_SIZE = 30
# 数据增强可以跟着训练情况进行修改



# 把图片载入近内存的方法，并且进行归一化处理
def load_image(path, filename, height, width, channels):
    images = []
    for image_name in filename:
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.resize(image, (height, width))
        images.append(image / 255.0)
    images = np.array(images)
    images = images.reshape([-1, height, width, channels])
    return images

# 对训练集进行分割，分割成训练集和验证集
# dateset = load_image(path, filesname, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
# test_size:设定验证集的占总数据集的百分比,0.2就是20%
# random_state: 随机因子,当这个值为固定某个值的时候,每次运行时切割的数据将是相同的,不设置则每次都不同(应该是这样)
def make_train_and_val_set(dataset, labels, test_size, random_state):
    train_set, val_set, train_label, val_label = train_test_split(dataset, labels,
                                                                  test_size=test_size,
                                                                  random_state=random_state)
    return train_set, val_set, train_label, val_label


# 根据类型建立标准化标签
encoder = preprocessing.LabelEncoder()
encoder.fit(labels)


# 将label转换成数字格式并且进行one_hot编码
def label2vec(labels):
    labels1 = encoder.transform(labels)
    one_hot_labels1 = keras.utils.to_categorical(labels1, num_classes=102)
    return labels1, one_hot_labels1


# 数据增强
# 开始的时候把先改成0.2训练，在无法上升的时候从0.2都改成0.1可以增加最后正确率
train_datagen = ImageDataGenerator(width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',)


# 使用预训练模型
def bulid_model():
    # 使用VGG19,DenseNet在数据集小的时候用处比较小,而且Vgg系列对图像的特征提取更优秀
    vgg19 = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet',
                                           input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3], classes=1000)
    model = Flatten()(vgg19.output)
    model = Dense(2048, activation='relu', name='fc1')(model)
    model = Dropout(0.4)(model)
    model = Dense(102, activation='softmax', name='output')(model)
    translate_model = Model(vgg19.input, model, name='vgg19')
    # 这里使用SGD的优化器,效果会比Adadelta效果高出5%左右
    translate_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(),
                            metrics=['accuracy'])
    translate_model.summary()
    return translate_model


model = bulid_model()

# 读取模型
# model = load_model('Vgg19.h5')


# 因为之前的学习率比较大 默认是0.01,在训练后期为了更为精准，调到0.001甚至0.0001，当然一开始使用0.001也是可以的
# model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.sgd(lr=0.001),
#                             metrics=['accuracy'])


# 保存模型
# model.save('Vgg19.h5')
#模型训练,2种方案,先使用第一种训练到正确率无法上升后使用第二种方案,可以提升1%左右,
#方案1,直接切割1次训练集进行训练到顶
# 先把图片名字和label进行划分,等需要读图的时候再去读
train_set_name, val_set_name, train_label, val_label = make_train_and_val_set(filesname, labels, 0.2, random.randint(10, 100))
# 获取验证集
val_set = load_image(path, val_set_name, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
val_label0, val_label1 = label2vec(val_label)
#获取训练集
train_set = load_image(path, train_set_name, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
train_label0, train_label1 = label2vec(train_label)

#进行训练
model.fit_generator(train_datagen.flow(train_set, train_label1, batch_size=30), steps_per_epoch=len(train_set)//30,
                    epochs=EPOCHS_SIZE, validation_data=(val_set, val_label1))
###################方案1训练完毕
# 第二种方案,进行多次随机分割训练集进行训练
# for i in range(4):
#     #这里就是每次都生成不同的随机因子来保证切割的不重复
#     train_set_name, val_set_name, train_label, val_label = make_train_and_val_set(filesname, labels, 0.2,
#                                                                                   random.randint(10, 100))
#     val_set = []
#     train_set = []
#     val_set = load_image(path, val_set_name, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
#     val_label0, val_label1 = label2vec(val_label)
#     # 获取训练集
#     train_set = load_image(path, train_set_name, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
#     train_label0, train_label1 = label2vec(train_label)
#     #batch_siz不能小于16,小于16可能正确率就无法提高了,最高我是能到40,经过测试是越高效果越好,但不是绝对的
#     model.fit_generator(train_datagen.flow(train_set, train_label1, batch_size=30), steps_per_epoch=len(train_set)//30
#     ,epochs=2, validation_data=(val_set, val_label1))
#     val_set = []
#     train_set = []


# 获取测试集
test_file_name = []
for root, dirs, files in os.walk(test_path):
    for filename in files:
        test_file_name.append(filename)
test_set = load_image(test_path, test_file_name, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

# 进行预测
preds = model.predict(test_set)


# 把类型转回成类型名称
def vec2label(label_vel):
    label = encoder.inverse_transform(label_vel)
    return label


# topK
def get_top_k_label(perds, k=1):
    top_k = tf.nn.top_k(perds, k).indices
    with tf.Session() as sess:
        top_k = sess.run(top_k)
        print(top_k)
    return top_k


result = get_top_k_label(preds, 5)
# 输出
sub = pd.DataFrame()
sub['ID'] = test_file_name
sub['result1'] = vec2label(result[:, 0])
sub['result2'] = vec2label(result[:, 1])
sub['result3'] = vec2label(result[:, 2])
sub['result4'] = vec2label(result[:, 3])
sub['result5'] = vec2label(result[:, 4])
sub.to_csv(export_file, index=False)
