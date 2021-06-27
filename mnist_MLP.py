
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import Counter

#预处理
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data() 
#print(X_train.shape, y_train.shape)   # 输出训练集样本和标签的大小
#print(X_train[30])
print(X_train.shape)
totalpx=X_train.shape[1]*X_train.shape[2]
X_train=X_train.reshape(X_train.shape[0],totalpx).astype('float32')
X_test=X_test.reshape(X_test.shape[0],totalpx).astype('float32')
label_cnt = Counter(y_train)
typenum=len(label_cnt)
#归一化
X_train = X_train / 255
X_test = X_test / 255

y_train=tf.keras.utils.to_categorical(y_train,typenum)
y_test=tf.keras.utils.to_categorical(y_test,typenum)

def network():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(512,input_shape=(totalpx,),activation='relu'))
    model.add(tf.keras.layers.Dense(512,input_shape=(512,),activation='relu'))
    model.add(tf.keras.layers.Dense(typenum,activation='softmax'))
    model.summary()  # 查看模型架构

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  # 定义模型训练细节，包括交叉熵损失函数，Adam优化器和准确率评价指标
    return model

model=network()
model.fit(X_train, y_train, batch_size=128,epochs=5, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'测试集损失值: {test_loss}, 测试集准确率: {test_acc}')


