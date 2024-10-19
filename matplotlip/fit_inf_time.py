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

# CICIDS-2017
DCNN_Fitting_Time_Multi = np.load('D:/results/CICIDS-2017/history/[CICIDS2017] [CNN] Multiclass classification Fitting time.npy',allow_pickle='TRUE').tolist()
DCNN_Inference_Time_Multi = np.load('D:/results/CICIDS-2017/history/[CICIDS2017] [CNN] Multiclass classification Inference time.npy',allow_pickle='TRUE').tolist()

DCNN_Fitting_Time_Binary = np.load('D:/results/CICIDS-2017/history/[CICIDS2017] [CNN] Binary classification Fitting time.npy',allow_pickle='TRUE').tolist()
DCNN_Inference_Time_Binary = np.load('D:/results/CICIDS-2017/history/[CICIDS2017] [CNN] Binary classification Inference time.npy',allow_pickle='TRUE').tolist()

DCNN_Fitting_Time_Multi
DCNN_Inference_Time_Multi

DCNN_Fitting_Time_Binary
DCNN_Inference_Time_Binary

DNN_Fitting_Time_Multi = np.load('D:/results/CICIDS-2017/history/[CICIDS2017] [DNN] Multiclass classification Fitting time.npy',allow_pickle='TRUE').tolist()
DNN_Inference_Time_Multi = np.load('D:/results/CICIDS-2017/history/[CICIDS2017] [DNN] Multiclass classification Inference time.npy',allow_pickle='TRUE').tolist()

DNN_Fitting_Time_Binary = np.load('D:/results/CICIDS-2017/history/[CICIDS2017] [DNN] Binary classification Fitting time.npy',allow_pickle='TRUE').tolist()
DNN_Inference_Time_Binary = np.load('D:/results/CICIDS-2017/history/[CICIDS2017] [DNN] Binary classification Inference time.npy',allow_pickle='TRUE').tolist()

DNN_Fitting_Time_Multi
DNN_Inference_Time_Multi

DNN_Fitting_Time_Binary
DNN_Inference_Time_Binary

# CICIoMT2024
DCNN_Fitting_Time_Multi = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [CNN] Multiclass classification Fitting time.npy',allow_pickle='TRUE').tolist()
DCNN_Inference_Time_Multi = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [CNN] Multiclass classification Inference time.npy',allow_pickle='TRUE').tolist()

DCNN_Fitting_Time_Binary = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [CNN] Binary classification Fitting time.npy',allow_pickle='TRUE').tolist()
DCNN_Inference_Time_Binary = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [CNN] Binary classification Inference time.npy',allow_pickle='TRUE').tolist()

DCNN_Fitting_Time_Multi
DCNN_Inference_Time_Multi

DCNN_Fitting_Time_Binary
DCNN_Inference_Time_Binary

# CICIoMT2024
DNN_Fitting_Time_Multi = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [DNN] Multiclass classification Fitting time.npy',allow_pickle='TRUE').tolist()
DNN_Inference_Time_Multi = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [DNN] Multiclass classification Inference time.npy',allow_pickle='TRUE').tolist()

DNN_Fitting_Time_Binary = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [DNN] Binary classification Fitting time.npy',allow_pickle='TRUE').tolist()
DNN_Inference_Time_Binary = np.load('D:/results/CICIoMT2024/history/[CICIoMT2024] [DNN] Binary classification Inference time.npy',allow_pickle='TRUE').tolist()

DNN_Fitting_Time_Multi
DNN_Inference_Time_Multi

DNN_Fitting_Time_Binary
DNN_Inference_Time_Binary





















