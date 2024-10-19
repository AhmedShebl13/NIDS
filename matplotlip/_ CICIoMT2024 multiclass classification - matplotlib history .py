import os
import numpy as np
import pandas as pd

from unidecode import unidecode

import tensorflow as tf
from tensorflow import keras

import pickle # saving and loading trained model
from os import path

# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# importing library for plotting
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import metrics
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report # for generating a classification report of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

from keras.layers import Dropout, Activation, Flatten, Convolution1D, Dropout, Reshape
from keras.layers import Dense # importing dense layer
from keras.models import Sequential #importing Sequential layer

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#     print(gpu, "\n")
# else:
#   print("No GPU device found")

import pandas as pd
import glob


history_cnn = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [CNN] Multiclass classification history.npy',allow_pickle='TRUE').item()
history_dnn = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [DNN] Multiclass classification history.npy',allow_pickle='TRUE').item()


# CNN Plot for training and validation loss
cnn_history_dict = history_cnn
cnn_loss_values = cnn_history_dict['loss']
cnn_val_loss_values = cnn_history_dict['val_loss']
cnn_acc = cnn_history_dict['accuracy']
cnn_val_acc = cnn_history_dict['val_accuracy']
cnn_start_by_epoch = 1
cnn_epochs = range(cnn_start_by_epoch, len(cnn_loss_values) + 1)


# DNN Plot for training and validation loss
dnn_history_dict = history_dnn
dnn_loss_values = dnn_history_dict['loss']
dnn_val_loss_values = dnn_history_dict['val_loss']
dnn_acc = dnn_history_dict['accuracy']
dnn_val_acc = dnn_history_dict['val_accuracy']
dnn_start_by_epoch = 1
dnn_epochs = range(dnn_start_by_epoch, len(dnn_loss_values) + 1)



# Exploratory data analysis
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

plt.figure(figsize=(5,4))
plt.rc('font', size=7)          # controls default text sizes
plt.rc('axes', titlesize=7)     # fontsize of the axes title
plt.plot(cnn_epochs, cnn_acc[cnn_start_by_epoch-1:], label='CNN training accuracy', marker='x', markersize=2.5, mew=0.4, linewidth=0.8)
plt.plot(cnn_epochs, cnn_val_acc[cnn_start_by_epoch-1:], label='CNN validation accuracy', marker='x', markersize=2.5, mew=0.4, linewidth=0.8)
plt.plot(dnn_epochs, dnn_acc[dnn_start_by_epoch-1:], label='DNN training accuracy', linestyle = "dashdot", marker='+', markersize=2.5, mew=0.4, linewidth=0.8)
plt.plot(dnn_epochs, dnn_val_acc[dnn_start_by_epoch-1:], label='DNN validation accuracy', linestyle = "dashdot", marker='+', markersize=2.5, mew=0.4, linewidth=0.8)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Accuracy: DCNN', 'Validation Accuracy: DCNN','Accuracy: DNN', 'Validation Accuracy: DNN'], loc='lower right')
plt.grid(axis='x', which='major', color='#CCC', alpha= 0.25, linewidth=0.7)
plt.grid(axis='y', which='major', color='#CCC', linewidth=0.7)
ax = plt.gca()
plt.tight_layout()
plt.tick_params(left = False) 
plt.tick_params(bottom = False) 
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0.99,1)
plt.savefig('D:/results/CICIoMT2024/figures/[CICIoMT2024]_Multiclass_classification_accuracy.eps', format='eps', dpi=1200)
plt.show()
plt.clf()

plt.figure(figsize=(5,4))
plt.rc('font', size=7)          # controls default text sizes
plt.rc('axes', titlesize=7)     # fontsize of the axes title
plt.plot(cnn_epochs, cnn_loss_values[cnn_start_by_epoch-1:], label='CNN training Loss', marker='x', markersize=2.5, mew=0.4, linewidth=0.8)
plt.plot(cnn_epochs, cnn_val_loss_values[cnn_start_by_epoch-1:], label='CNN validation Loss', marker='x', markersize=2.5, mew=0.4, linewidth=0.8)
plt.plot(dnn_epochs, dnn_loss_values[dnn_start_by_epoch-1:], label='DNN training Loss', linestyle = "dashdot", marker='+', markersize=2.5, mew=0.4, linewidth=0.8)
plt.plot(dnn_epochs, dnn_val_loss_values[dnn_start_by_epoch-1:], label='DNN validation Loss', linestyle = "dashdot", marker='+', markersize=2.5, mew=0.4, linewidth=0.8)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss: DCNN', 'Validation Loss: DCNN','Loss: DNN', 'Validation Loss: DNN'], loc='upper right')
plt.grid(axis='x', which='major', color='#CCC', alpha= 0.25, linewidth=0.7)
plt.grid(axis='y', which='major', color='#CCC', linewidth=0.7)
ax = plt.gca()
plt.tight_layout()
plt.tick_params(left = False) 
plt.tick_params(bottom = False) 
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(-0.002,0.05)
plt.savefig('D:/results/CICIoMT2024/figures/[CICIoMT2024]_Multiclass_classification_loss.eps', format='eps', dpi=1200)
plt.show()
plt.clf()


















