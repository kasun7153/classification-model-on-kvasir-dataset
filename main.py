#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy.random import seed
seed(888)

import os
import numpy as np
import tensorflow as tf
tf.random.set_seed(404)
import itertools

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from IPython.display import display
from keras.preprocessing.image import array_to_img
from keras.callbacks import TensorBoard
from keras.preprocessing.image import img_to_array,load_img

from time import strftime

from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Constants
# 

# In[323]:


LOG_DIR = 'tensorboard_cifar_logs/'
LABEL_NAMES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
TOTAL_INPUTS = IMAGE_PIXELS * COLOR_CHANNELS
NR_CLASSES = 8


# In[324]:


y_train_all=[]
x_train_all=[]

for i in range (0,8):
    file_arr = os.listdir(f'tranning-dataset/{LABEL_NAMES[i]}')
    for j in file_arr:
        file_type=[i]
        y_train_all.append(file_type)
        path=f'tranning-dataset/{LABEL_NAMES[i]}/{j}'
        pic = load_img(path, target_size=(IMAGE_WIDTH,IMAGE_HEIGHT))
        pic_array=img_to_array(pic)
        x_train_all.append(pic_array)
    
x_train_all=np.array(x_train_all,dtype="uint8")   
y_train_all=np.array(y_train_all,dtype="uint8")   


# In[325]:


y_test=[]
x_test=[]

for i in range (0,8):
    file_arr = os.listdir(f'testing-dataset/{LABEL_NAMES[i]}')
    for j in file_arr:
        file_type=[i]
        y_test.append(file_type)
        path=f'testing-dataset/{LABEL_NAMES[i]}/{j}'
        pic = load_img(path, target_size=(IMAGE_WIDTH,IMAGE_HEIGHT))
        pic_array=img_to_array(pic)
        x_test.append(pic_array)
    
x_test=np.array(x_test,dtype="uint8")   
y_test=np.array(y_test,dtype="uint8")  
        


# In[326]:


y_validation=[]
x_validation=[]

for i in range (0,8):
    file_arr = os.listdir(f'validation-dataset/{LABEL_NAMES[i]}')
    for j in file_arr:
        file_type=[i]
        y_validation.append(file_type)
        path=f'validation-dataset/{LABEL_NAMES[i]}/{j}'
        pic = load_img(path, target_size=(IMAGE_WIDTH,IMAGE_HEIGHT))
        pic_array=img_to_array(pic)
        x_validation.append(pic_array)
    
x_validation=np.array(x_validation,dtype="uint8")   
y_validation=np.array(y_validation,dtype="uint8")  


# In[327]:


plt.imshow(x_train_all[0])
plt.xlabel(LABEL_NAMES[y_train_all[4][0]], fontsize=15)
plt.show()


# In[328]:


pic = array_to_img(x_train_all[0])
display(pic)


# In[329]:


type(x_train_all[0][0][0][0])


# ## Pre Process data

# In[330]:


x_train_all, x_test,x_validation = x_train_all / 255.0, x_test / 255.0,x_validation/ 255.0


# In[331]:


x_train_all[0][0][0][0]


# In[332]:


x_train_all = x_train_all.reshape(x_train_all.shape[0], TOTAL_INPUTS)
x_test = x_test.reshape(len(x_test), TOTAL_INPUTS)
x_validation = x_validation.reshape(len(x_validation), TOTAL_INPUTS)


# In[ ]:





# ## Define the Neural Network using Keras

# In[333]:


model_2 = Sequential()
model_2.add(Dropout(0.2, seed=42, input_shape=(TOTAL_INPUTS,)))
model_2.add(Dense(1000, activation='relu', name='m2_hidden1'))
model_2.add(Dense(800, activation='relu', name='m2_hidden2'))
model_2.add(Dense(400, activation='relu', name='m2_hidden3'))
model_2.add(Dense(200, activation='relu', name='m2_hidden11'))
model_2.add(Dense(100, activation='relu', name='m2_hidden4'))
model_2.add(Dense(64, activation='relu', name='m2_hidden5'))
model_2.add(Dense(15, activation='relu', name='m2_hidden6'))
model_2.add(Dense(10, activation='relu', name='m2_hidden7'))
model_2.add(Dense(8, activation='softmax', name='m2_output'))

model_2.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])


# ## Tensorboard (visualising learning)

# In[334]:


def get_tensorboard(model_name):

    folder_name = f'{model_name} at {strftime("%H %M")}'
    dir_paths = os.path.join(LOG_DIR, folder_name)

    try:
        os.makedirs(dir_paths)
    except OSError as err:
        print(err.strerror)
    else:
        print('Successfully created directory')

    return TensorBoard(log_dir=dir_paths)


# ## Fit the Model

# In[335]:


samples_per_batch = 128


# In[336]:


get_ipython().run_cell_magic('time', '', "nr_epochs = 100\nmodel_2.fit(x_train_all, y_train_all, batch_size=samples_per_batch, epochs=nr_epochs,\n            callbacks=[get_tensorboard('Model 1 XL')],verbose=0,validation_data=(x_validation, y_validation))")


# ## Predictions on Individual Images

# In[337]:


np.set_printoptions(precision=3)


# In[338]:


test_img = np.expand_dims(x_test[12], axis=0)
predicted_val = model_2.predict_classes(test_img)[0]
print(f'Actual value: {y_test[10][0]} vs. predicted: {predicted_val}')


# In[339]:


pic = load_img("1.jpg", target_size=(IMAGE_WIDTH,IMAGE_HEIGHT))

solo_test=[]
pic_array=img_to_array(pic)
solo_test.append(pic_array)
    
solo_test=np.array(solo_test,dtype="uint8")  

solo_test= solo_test / 255.0
solo_test = solo_test.reshape(solo_test.shape[0], TOTAL_INPUTS)
test_img = np.expand_dims(solo_test[0], axis=0)
predicted_val = model_2.predict_classes(test_img)[0]
print(f'predicted: {LABEL_NAMES[predicted_val]}')


# # Evaluation

# In[340]:


test_loss, test_accuracy = model_2.evaluate(x_test, y_test)
print(f'Test loss is {test_loss:0.3} and test accuracy is {test_accuracy:0.1%}')


# ## Confusion Matrix

# In[341]:


predictions = model_2.predict_classes(x_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)


# In[342]:


nr_rows = conf_matrix.shape[0]
nr_cols = conf_matrix.shape[1]


# In[343]:


plt.figure(figsize=(7,7), dpi=95)
plt.imshow(conf_matrix, cmap=plt.cm.Greens)

plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Labels', fontsize=12)
plt.xlabel('Predicted Labels', fontsize=12)

tick_marks = np.arange(NR_CLASSES)
plt.yticks(tick_marks, LABEL_NAMES)
plt.xticks(tick_marks, LABEL_NAMES)

plt.colorbar()

for i, j in itertools.product(range(nr_rows), range(nr_cols)):
    plt.text(j, i, conf_matrix[i, j], horizontalalignment='center',
            color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')
    

plt.show()
