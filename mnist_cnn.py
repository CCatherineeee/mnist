import numpy as np
import tensorflow as tf
import numpy as np

from collections import Counter
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


#预处理
(X_train, y_train), (X_test, y_test) =tf.keras.datasets.mnist.load_data() 

#print(X_train[30])
totalpx=X_train.shape[1]*X_train.shape[2]
X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')

#X_train = np.expand_dims(X_train, axis=3)
#X_test = np.expand_dims(X_test, axis=3)

print(X_train.shape, X_test.shape)   # 输出训练集样本和标签的大小

#归一化
X_train = X_train / 255
X_test = X_test / 255


#print(y_test)

y_train=tf.keras.utils.to_categorical(y_train)
y_test=tf.keras.utils.to_categorical(y_test)

numtype=y_test[1]
#print(y_test)


def network():
    model=tf.keras.models.Sequential()
    #卷积层
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,input_shape=(28,28,1),activation='relu'))
    #池化层
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))  
    #压平
    model.add(tf.keras.layers.Flatten())
    #全连接层
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model=network()

h=model.fit(X_train, y_train, batch_size=64,epochs=10, validation_data=(X_test, y_test))
test_loss,test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'测试集损失值: {test_loss}, 测试集准确率: {test_acc}')

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

 
print(h.history.keys()) 
accuracy = h.history['accuracy']
val_accuracy = h.history['val_accuracy']
loss = h.history['loss']
val_loss = h.history['val_loss']
epochs = range(len(accuracy))

plt.figure()
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'bo', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy_1.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss_1.png', bbox_inches='tight', dpi=300)

model_path="E:/神经网络/mnist/"
model.save(model_path)
print("saved")