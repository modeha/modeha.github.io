---
layout: post
title: Handwritten Text Recognition
date: 2025-01-25 13:06 -0500
---
## **Handwritten Text Recognition**

---
Handwritten Text Recognition with Tensorflow2 & Keras & IAM Dataset.

Convolutional Recurrent Neural Network. CTC.

Author : Mohsen Dehghani



-






## Dataset used:


Data: [IAM Dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database)

Used in this project: [words.tgz](http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/)



```python
import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMNotebookCallback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```

    Using TensorFlow backend.
    


```python

```


```python
import tensorflow as tf

#ignore warnings in the output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
```


```python
from tensorflow.python.client import device_lib

# Check all available devices if GPU is available
print(device_lib.list_local_devices())
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 94618472378254487
    , name: "/device:XLA_CPU:0"
    device_type: "XLA_CPU"
    memory_limit: 17179869184
    locality {
    }
    incarnation: 17787937145416098544
    physical_device_desc: "device: XLA_CPU device"
    , name: "/device:XLA_GPU:0"
    device_type: "XLA_GPU"
    memory_limit: 17179869184
    locality {
    }
    incarnation: 15991697771353570040
    physical_device_desc: "device: XLA_GPU device"
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 11150726272
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 14111058102008547075
    physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"
    ]
    Device mapping:
    /job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
    /job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
    /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7
    
    


```python
tf.config.experimental.list_physical_devices('GPU')
```




    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]




```python
from google.colab import drive
drive.mount('/content/gdrive')
```


```python
with open('./words.txt') as f:
    contents = f.readlines()

lines = [line.strip() for line in contents] 
lines[0]
```




    'a01-000u-00-00 ok 154 408 768 27 51 AT A'




```python
max_label_len = 0

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 

# string.ascii_letters + string.digits (Chars & Digits)
# or 
# "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

print(char_list, len(char_list))

def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, chara in enumerate(txt):
        dig_lst.append(char_list.index(chara))
        
    return dig_lst
```

    !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 78
    


```python
images = []
labels = []

RECORDS_COUNT = 10000
```


```python
train_images = []
train_labels = []
train_input_length = []
train_label_length = []
train_original_text = []

valid_images = []
valid_labels = []
valid_input_length = []
valid_label_length = []
valid_original_text = []

inputs_length = []
labels_length = []
```


```python
def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    
    img = img.astype('float32')
    
    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
        
    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)
    
    img = cv2.subtract(255, img)
    
    img = np.expand_dims(img, axis=2)
    
    # Normalize 
    img = img / 255
    
    return img
```

## Generate train & validation set


```python
for index, line in enumerate(lines):
    splits = line.split(' ')
    status = splits[1]
    
    if status == 'ok':
        word_id = splits[0]
        word = "".join(splits[8:])
        
        splits_id = word_id.split('-')
        filepath = 'words/{}/{}-{}/{}.png'.format(splits_id[0], 
                                                  splits_id[0], 
                                                  splits_id[1], 
                                                  word_id)
        
        # process image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        try:
            img = process_image(img)
        except:
            continue
            
        # process label
        try:
            label = encode_to_labels(word)
        except:
            continue
        
        if index % 10 == 0:
            valid_images.append(img)
            valid_labels.append(label)
            valid_input_length.append(31)
            valid_label_length.append(len(word))
            valid_original_text.append(word)
        else:
            train_images.append(img)
            train_labels.append(label)
            train_input_length.append(31)
            train_label_length.append(len(word))
            train_original_text.append(word)
        
        if len(word) > max_label_len:
            max_label_len = len(word)
    
    if index >= RECORDS_COUNT:
        break
```


```python
train_padded_label = pad_sequences(train_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))

valid_padded_label = pad_sequences(valid_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))
```


```python
train_padded_label.shape, valid_padded_label.shape
```




    ((7850, 16), (876, 16))



## Converts to Numpy array


```python
train_images = np.asarray(train_images)
train_input_length = np.asarray(train_input_length)
train_label_length = np.asarray(train_label_length)

valid_images = np.asarray(valid_images)
valid_input_length = np.asarray(valid_input_length)
valid_label_length = np.asarray(valid_label_length)
```


```python
train_images.shape
```




    (7850, 32, 128, 1)



## Build Model
Convolutional Recurrent Neural Network 


```python
# input with shape of height=32 and width=128 
inputs = Input(shape=(32,128,1))
 
# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 
# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)
```


```python
act_model.summary()
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 32, 128, 1)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 32, 128, 64)       640       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 16, 64, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 16, 64, 128)       73856     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 8, 32, 128)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 8, 32, 256)        295168    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 8, 32, 256)        590080    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 4, 32, 256)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 4, 32, 512)        1180160   
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 4, 32, 512)        2048      
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 4, 32, 512)        2359808   
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 4, 32, 512)        2048      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 2, 32, 512)        0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 1, 31, 512)        1049088   
    _________________________________________________________________
    lambda_1 (Lambda)            (None, 31, 512)           0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 31, 512)           1574912   
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, 31, 512)           1574912   
    _________________________________________________________________
    dense_1 (Dense)              (None, 31, 79)            40527     
    =================================================================
    Total params: 8,743,247
    Trainable params: 8,741,199
    Non-trainable params: 2,048
    _________________________________________________________________
    


```python
the_labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, the_labels, input_length, label_length])

#model to be used at training time
model = Model(inputs=[inputs, the_labels, input_length, label_length], outputs=loss_out)
```


```python
batch_size = 8
epochs = 60
e = str(epochs)
optimizer_name = 'adam'
```


```python
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = optimizer_name, metrics=['accuracy'])

filepath="{}o-{}r-{}e-{}t-{}v.hdf5".format(optimizer_name,
                                          str(RECORDS_COUNT),
                                          str(epochs),
                                          str(train_images.shape[0]),
                                          str(valid_images.shape[0]))

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
```


```python
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```


```python
history = model.fit(x=[train_images, train_padded_label, train_input_length, train_label_length],
                    y=np.zeros(len(train_images)),
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_data=([valid_images, valid_padded_label, valid_input_length, valid_label_length], [np.zeros(len(valid_images))]),
                    verbose=1,callbacks=callbacks_list)
```

## Training Accuracy


```python
# predict outputs on validation images
prediction = act_model.predict(train_images[150:170])
 
# use CTC decoder
decoded = K.ctc_decode(prediction,   
                       input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                       greedy=True)[0][0]

out = K.get_value(decoded)

# see the results
for i, x in enumerate(out):
    print("original_text =  ", train_original_text[150+i])
    print("predicted text = ", end = '')
    for p in x:
        if int(p) != -1:
            print(char_list[int(p)], end = '')
    plt.imshow(train_images[150+i].reshape(32,128), cmap=plt.cm.gray)
    plt.show()
    print('\n')
```

    original_text =   large
    predicted text = large


    
![png](test10_files/test10_28_1.png)
    


    
    
    original_text =   majority
    predicted text = majority


    
![png](test10_files/test10_28_3.png)
    


    
    
    original_text =   of
    predicted text = of


    
![png](test10_files/test10_28_5.png)
    


    
    
    original_text =   Labour
    predicted text = Labour


    
![png](test10_files/test10_28_7.png)
    


    
    
    original_text =   MPs
    predicted text = MPs


    
![png](test10_files/test10_28_9.png)
    


    
    
    original_text =   are
    predicted text = are


    
![png](test10_files/test10_28_11.png)
    


    
    
    original_text =   to
    predicted text = to


    
![png](test10_files/test10_28_13.png)
    


    
    
    original_text =   turn
    predicted text = twn


    
![png](test10_files/test10_28_15.png)
    


    
    
    original_text =   down
    predicted text = down


    
![png](test10_files/test10_28_17.png)
    


    
    
    original_text =   the
    predicted text = the


    
![png](test10_files/test10_28_19.png)
    


    
    
    original_text =   Foot-
    predicted text = Foot-


    
![png](test10_files/test10_28_21.png)
    


    
    
    original_text =   be
    predicted text = be


    
![png](test10_files/test10_28_23.png)
    


    
    
    original_text =   that
    predicted text = that


    
![png](test10_files/test10_28_25.png)
    


    
    
    original_text =   as
    predicted text = as


    
![png](test10_files/test10_28_27.png)
    


    
    
    original_text =   Labour
    predicted text = Labour


    
![png](test10_files/test10_28_29.png)
    


    
    
    original_text =   MPs
    predicted text = MPs


    
![png](test10_files/test10_28_31.png)
    


    
    
    original_text =   opposed
    predicted text = opposed


    
![png](test10_files/test10_28_33.png)
    


    
    
    original_text =   the
    predicted text = the


    
![png](test10_files/test10_28_35.png)
    


    
    
    original_text =   Bill
    predicted text = Bill


    
![png](test10_files/test10_28_37.png)
    


    
    
    original_text =   which
    predicted text = which


    
![png](test10_files/test10_28_39.png)
    


    
    
    


```python
# plot accuracy and loss
def plotgraph(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
```


```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
```


```python
plotgraph(epochs, loss, val_loss)
```


    
![png](test10_files/test10_31_0.png)
    



```python
plotgraph(epochs, acc, val_acc)
```


    
![png](test10_files/test10_32_0.png)
    

