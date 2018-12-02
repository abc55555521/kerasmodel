from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
import cv2
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = 128
path = r'data\train_data'
a = pd.read_csv(r'data\train.csv')
filesname = a['filename']
labels = a['label']
encoder = LabelEncoder()
encoder.fit(labels)
#图片读取
def load_image(path,filesname,height,width,channels):
    images = []
    for image_name in filesname:
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.resize(image, (height, width))
        images.append(image / 255.0)
    images = np.array(images)
    images = images.reshape([-1,height,width,channels])
    return images
#one hot label
def label2vec(labels):

    labels1 = encoder.transform(labels)
    one_hot_labels1 = to_categorical(labels1, num_classes=102)
    return labels1,one_hot_labels1



# data_set = load_image(path,filesname,256,256,3)
#划分训练集和测试集
def make_train_and_val_set(dataset,labels,test_size):
    train_set,val_set,train_label,val_label = train_test_split(dataset,labels,
                                    test_size = test_size,random_state = 5)
    return train_set,val_set,train_label,val_label
train_set_name,val_set_name,train_label,val_label = make_train_and_val_set(filesname,labels,0.2)
#模型构建
def build_model():
    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3],
                     data_format='channels_last', name='conv1'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=2,name='pool1'))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', name='conv3'))
    model.add(MaxPooling2D(pool_size=2,name='pool2'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3, name='dropout2'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3, name='dropout3'))
    model.add(Dense(102, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    model.summary()
    return model
#数据增强
train_datagen = ImageDataGenerator(
        # rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
)

tb = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=False, # 是否可视化梯度直方图
                 write_images=False,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
callbacks = [tb]

#生成小批次数据训练
def data_generator(image_path,filesname,labels,batch_size):
    batches = (len(labels) + batch_size - 1) // batch_size


    while(True):
        for i in range(batches):
            y = labels[i*batch_size:(i+1)*batch_size]
            y0,y1 = label2vec(y)
            x_names = filesname[i*batch_size:(i+1)*batch_size]
            x = load_image(image_path,x_names,IMAGE_SIZE,IMAGE_SIZE,3)

            yield(x,y1)
print("加载图片")
#准备验证数据

val_set = load_image(path,val_set_name,IMAGE_SIZE,IMAGE_SIZE,3)
val_label0,val_label1 = label2vec(val_label)
BATCH_SIZE = 32
model = build_model()
print("训练开始")
train_set = load_image(path, train_set_name, IMAGE_SIZE, IMAGE_SIZE, 3)
train_label0, train_label1 = label2vec(train_label)
model.fit_generator(train_datagen.flow(train_set, train_label1, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(train_set) // BATCH_SIZE, epochs=30, validation_data=(val_set, val_label1),
          callbacks=callbacks)
# model.fit_generator(data_generator(path, train_set_name, train_label, BATCH_SIZE),
#                     (len(labels) + BATCH_SIZE - 1) // BATCH_SIZE, 1, validation_data=(val_set, val_label1))
print("训练结束，准备预测")
#保存模型
model.save('model.h5')
#model = keras.models.load_model('demo/model1.h5')

#one hot coding 转回字符串label
def vec2label(label_vec):

    label = encoder.inverse_transform(label_vec)
    return label
preds = model.predict(val_set)

test_path =  r'data\test_data'
b = pd.read_csv(r'data\test.csv')
t_filesname = b['filename']
# t_labels = b['label']
test_set = load_image(test_path, t_filesname, IMAGE_SIZE, IMAGE_SIZE, 3)
# print(len(test_set))
# # test_set *= 255
# t_labels0,t_labels1 = label2vec(t_labels)
preds1 = model.predict(test_set)
print (preds1)

def vec2label(label_vec):
    label = encoder.inverse_transform(label_vec)
    return label


def get_top_k_label(preds, k=1):
    top_k = tf.nn.top_k(preds, k).indices
    with tf.Session() as sess:
        top_k = sess.run(top_k)
    top_k_label = vec2label(top_k)
    return top_k_label


predslabel = get_top_k_label(preds1, 5)

submit = pd.DataFrame({'fliesname': t_filesname, 'pred1': predslabel[:, 0],
                       'pred2': predslabel[:, 1], 'pred3': predslabel[:, 2]
                          , 'pred4': predslabel[:, 3], 'pred5': predslabel[:, 4]})
submit.to_csv(r'data\submit.csv', index=False)
print("end")

