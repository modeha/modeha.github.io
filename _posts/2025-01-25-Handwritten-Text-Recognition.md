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

## Dataset used:

Used in this project: [IAM Dataset](http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/)



```python
import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMNotebookCallback
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Lambda, Dense, Bidirectional, LSTM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
with open('./data/words2.txt') as f:
    contents = f.readlines()

lines = [line.strip() for line in contents] 
lines[2],lines[20]
import random

# Define the portion (e.g., 10%)
portion = 1
num_samples = int(portion * len(lines))

# Randomly select 10% of the lines
lines_subset = random.sample(lines, num_samples)

lines = lines_subset
print(f"Original size: {len(lines)}, Subset size: {len(lines_subset)}")

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

def decode_to_labels(indices):
    """
    Decodes a list of indices back to the corresponding string using char_list.

    Args:
        indices (list of int): List of indices representing encoded characters.

    Returns:
        str: Decoded string.
    """
    decoded_str = ''.join([char_list[i] for i in indices])
    return decoded_str

# Example Usage
encoded_example = encode_to_labels("Hello123")
print("Encoded:", encoded_example)

decoded_example = decode_to_labels(encoded_example)
print("Decoded:", decoded_example)
images = []
labels = []

RECORDS_COUNT = 10000

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
        img = cv2.imread('./data/'+filepath, cv2.IMREAD_GRAYSCALE)
        
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
# Print subset shapes to verify
print("Subset shapes:", 
      len(train_images), 
      len(train_labels), 
      len(valid_images), 
      len(valid_labels))
train_padded_label = pad_sequences(train_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))

valid_padded_label = pad_sequences(valid_labels, 
                             maxlen=max_label_len, 
                             padding='post',
                             value=len(char_list))

train_images = np.asarray(train_images)
train_input_length = np.asarray(train_input_length)
train_label_length = np.asarray(train_label_length)

valid_images = np.asarray(valid_images)
valid_input_length = np.asarray(valid_input_length)
valid_label_length = np.asarray(valid_label_length)
train_images.shape,train_padded_label.shape,valid_images.shape, valid_padded_label.shape

```

    Original size: 115320, Subset size: 115320
    !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 78
    Encoded: [33, 56, 63, 63, 66, 14, 15, 16]
    Decoded: Hello123
    Subset shapes: 7579 7579 847 847
    

```python

# Define input shape (height=32, width=128)
inputs = Input(shape=(32, 128, 1))

# Convolutional layers
conv_1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(128, (3,3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(256, (3,3), activation='relu', padding='same')(pool_2)
conv_4 = Conv2D(256, (3,3), activation='relu', padding='same')(conv_3)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(512, (3,3), activation='relu', padding='same')(pool_4)
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(512, (3,3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2,2), activation='relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# Bidirectional LSTM layers
blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(blstm_1)

outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

# Model for prediction
act_model = Model(inputs, outputs)

# Inputs for training (additional)
the_labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# Define the custom CTC loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Add the CTC loss layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, the_labels, input_length, label_length])

# Training model with CTC loss
model = Model(inputs=[inputs, the_labels, input_length, label_length], outputs=loss_out)

# Compile the model
batch_size = 8
epochs = 60
optimizer_name = 'adam'
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer_name, metrics=['accuracy'])

# Callbacks setup
model_save_path = "./ocr_ctc_model"
checkpoint = ModelCheckpoint(
    filepath=model_save_path + '/best_model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)

plot_callback = TrainingPlot()
callbacks_list = [checkpoint, plot_callback]

# Save the model without the CTC loss for better reusability
model_without_ctc = Model(inputs=inputs, outputs=outputs)
model_without_ctc.save(model_save_path, save_format='tf')

print(f"Model saved successfully to {model_save_path}")

# Training with callbacks
history = model.fit(
    x=[train_images, train_padded_label, train_input_length, train_label_length],
    y=np.zeros(len(train_images)),
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=(
        [valid_images, valid_padded_label, valid_input_length, valid_label_length],
        [np.zeros(len(valid_images))]
    ),
    verbose=1,
    callbacks=callbacks_list
)

print("Training complete. Model saved successfully.")

```
<img src="/assets/figures/IAM_accuracy.png">
  


    948/948 [==============================] - 555s 585ms/step - loss: 3.0008 - accuracy: 0.3644 - val_loss: 5.7751 - val_accuracy: 0.3542
    Training complete. Model saved successfully.
    


```python
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint


# Load the saved model without CTC for inference or re-training
loaded_model = load_model(model_save_path)

print("Model loaded successfully.")

# Re-add the CTC loss Lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [loaded_model.output, the_labels, input_length, label_length]
)

# Build the complete model again with CTC loss
final_model = Model(inputs=[loaded_model.input, the_labels, input_length, label_length], outputs=loss_out)

print("CTC layer re-added successfully.")

# Compile the model again with the same optimizer
final_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer_name, metrics=['accuracy'])

# Define Callbacks for Resuming Training
checkpoint = ModelCheckpoint(
    filepath="./ocr_ctc_model/best_model_resume.h5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)


# Callbacks list
callbacks_list = [checkpoint, plot_callback]

# Continue training from the last saved epoch (starting from epoch 3)
initial_epoch = 2  # Training previously done for 2 epochs

history = final_model.fit(
    x=[train_images, train_padded_label, train_input_length, train_label_length],
    y=np.zeros(len(train_images)),
    batch_size=batch_size, 
    epochs=5,  # Continue training till epoch 5
    initial_epoch=initial_epoch,  # Start from epoch 3
    validation_data=(
        [valid_images, valid_padded_label, valid_input_length, valid_label_length],
        [np.zeros(len(valid_images))]
    ),
    verbose=1,
    callbacks=callbacks_list  # Including checkpointing and plotting
)

print("Training resumed and completed successfully.")

```

## Training Accuracy


```python
# predict outputs on validation images
m=1
prediction = act_model.predict(train_images[m:2])
 
# use CTC decoder
decoded = K.ctc_decode(prediction,   
                       input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                       greedy=True)[0][0]

out = K.get_value(decoded)

# see the results
for i, x in enumerate(out):
    print("original_text =  ", train_original_text[m+i])
    print("predicted text = ", end = '')
    for p in x:
        if int(p) != -1:
            print(char_list[int(p)], end = '')
    plt.imshow(train_images[m+i].reshape(32,128), cmap=plt.cm.gray)
    plt.show()
    print('\n')
```

    1/1 [==============================] - 0s 169ms/step
    original_text =   taken
    predicted text = taken


<img src="/assets/figures/IAM_accuracy2.png">


    


    
    
    
