import os
import glob
import time
import numpy as np
import pandas as pd
import seaborn as sn
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt

from os import path
from tensorflow import keras
from matplotlib.pyplot import *
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, Flatten, Convolution1D, UpSampling1D, MaxPooling1D
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, multilabel_confusion_matrix

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#     print(gpu, "\n")
# else:
#   print("No GPU device found")

# full dataset
path = r'/Users/ahmad/Documents/Master/Datasets/CICIoT2023' 
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, encoding='cp1252', index_col=None, header=0)
    li.append(df)
    print("Read Completed for ", filename)
df = pd.concat(li, axis=0, ignore_index=True)
df.describe()
df.head()
df.info()

print(df.columns)
print(df["label"].value_counts())
print(df.shape)

Attack = ['DDoS-ACK_Fragmentation',     
          'DDoS-UDP_Flood',       
          'DDoS-SlowLoris',               
          'DDoS-ICMP_Flood',        
          'DDoS-RSTFINFlood',    
          'DDoS-PSHACK_Flood',        
          'DDoS-HTTP_Flood',              
          'DDoS-UDP_Fragmentation',     
          'DDoS-TCP_Flood',  
          'DDoS-SYN_Flood',        
          'DDoS-SynonymousIP_Flood',    
          'DDoS-ICMP_Fragmentation',
          'DoS-UDP_Flood',    
          'DoS-TCP_Flood',    
          'DoS-SYN_Flood',    
          'DoS-HTTP_Flood',
          'MITM-ArpSpoofing',     
          'DNS_Spoofing',
          'DictionaryBruteForce',
          'Recon-HostDiscovery', 
          'Recon-OSScan',     
          'Recon-PortScan',               
          'Recon-PingSweep',         
          'VulnerabilityScan',
          'SqlInjection',        
          'BrowserHijacking',         
          'CommandInjection',       
          'XSS',         
          'Backdoor_Malware',          
          'Uploading_Attack',
          'Mirai-greeth_flood',     
          'Mirai-udpplain',     
          'Mirai-greip_flood']


def classify_attacks(data):
    data['label'].replace(Attack,'Attack',inplace=True)
classify_attacks(df)

print(df["label"].value_counts())
print(df.shape)

data_clean = df.dropna().reset_index()
data_clean.drop_duplicates(keep='first', inplace = True)
data_clean['label'].value_counts()
print("Read {} rows.".format(len(data_clean)))

data_clean.columns
data_clean = data_clean.dropna().reset_index()
labelencoder = LabelEncoder()
data_clean['label'] = labelencoder.fit_transform(data_clean['label'])
data_clean['label'].value_counts()
print(data_clean.shape)
print(data_clean['label'].value_counts())

data_np = data_clean.to_numpy(dtype="float32")
data_np = data_np[~np.isinf(data_np).any(axis=1)]
X = data_np[:, :-1]
enc = OneHotEncoder()
Y = enc.fit_transform(data_np[:, -1].reshape(-1, 1)).toarray()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.1, random_state=33, shuffle=True)

_features = X.shape[1]
n_classes = Y.shape[1]

# DNN model
# Convolutional Encoder
input_img = Input(shape=(_features))

fc1 = Dense(128, activation='relu')(input_img)
do1 = Dropout(0.1)(fc1)

fc2 = Dense(256, activation='relu')(do1)
do2 = Dropout(0.1)(fc2)

fc3 = Dense(128, activation='relu')(do2)
do3 = Dropout(0.1)(fc3)

fc4 = Dense(n_classes, kernel_initializer='normal')(do3)

softmax = Dense(n_classes, activation='sigmoid', name='classification')(fc4)

model = Model(inputs=input_img, outputs=softmax)
model.summary()

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                              batch_size=128,
                              epochs=30,
                              verbose=True,
                              validation_data=(X_test, Y_test))    

np.save('/Users/ahmad/Documents/Master/results/CICIoT2023/History/[CICIoT2023] [DNN] Binary classification history.npy',history.history)

# Plot for training and validation loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
start_by_epoch = 1
epochs = range(start_by_epoch, len(loss_values) + 1)


# Exploratory data analysis
plt.plot(epochs, acc[start_by_epoch-1:], label='Training accuracy', marker='x', markersize=2.5, mew=0.4, linewidth=0.8)
plt.plot(epochs, val_acc[start_by_epoch-1:], label='Validation accuracy', marker='x', markersize=2.5, mew=0.4, linewidth=0.8)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.grid()
ax = plt.gca()
plt.tight_layout()
ax.spines['bottom'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0.992,1)
plt.savefig('/Users/ahmad/Documents/Master/results/CICIoT2023/figuers/[CICIoT2023] [DNN] Binary classification accuracy.eps', format='eps', dpi=1200)
plt.show()
plt.clf()

plt.plot(epochs, loss_values[start_by_epoch-1:], label='Training Loss', marker='x', markersize=2.5, mew=0.4, linewidth=0.8)
plt.plot(epochs, val_loss_values[start_by_epoch-1:], label='Validation Loss', marker='x', markersize=2.5, mew=0.4, linewidth=0.8)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.grid()
ax = plt.gca()
plt.tight_layout()
ax.spines['bottom'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0,0.02)
plt.savefig('/Users/ahmad/Documents/Master/results/CICIoT2023/figuers/[CICIoT2023] [DNN] Binary classification loss.eps', format='eps', dpi=1200)
plt.show()
plt.clf()

pred = model.predict(X_test)

# Convert predictions classes to one hot vectors 
pred = np.argmax(pred,axis=1)
# Convert validation observations to one hot vectors
y_test = Y_test.argmax(axis=1)

print(pred.shape)
print(y_test.shape)

score = metrics.accuracy_score(y_test, pred)
rscore = recall_score(y_test, pred, average='weighted')
ascore = precision_score(y_test, pred, average='weighted')
ascore_macro = precision_score(y_test, pred, average='macro')
f1score= f1_score(y_test, pred, average='weighted') #F1 = 2 * (precision * recall) / (precision + recall) for manual

print("Validation score: {}".format(score))
print("Recall score: {}".format(rscore))
print("Precision score: {}".format(ascore))
print("Precision score macro: {}".format(ascore_macro))
print("F1 Measure score: {}".format(f1score))

confMat = confusion_matrix(y_test, pred)
print(confMat)
cm_df = pd.DataFrame(confMat)

labels = ['BenignTraffic', 'Attack']

plt.figure(figsize=(25,10))
sn.set(font_scale=2.4)
sn.heatmap(cm_df, annot=True, annot_kws={"size":22}, fmt='g', xticklabels=labels, yticklabels=labels, cmap='Blues')

#sn.heatmap(cm_df, annot=True, annot_kws={"size":12}, fmt='g', xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')   
plt.savefig('/Users/ahmad/Documents/Master/results/CICIoT2023/figuers/[CICIoT2023] [DNN] Binary classification confusion matrix.eps',  bbox_inches="tight", format='eps', dpi=1200) 
plt.show() 
# plt.figure(figsize = (30,25))
# sn.set(font_scale=2.4)
# sn.heatmap(confMat, annot=True, fmt='.2f',cmap='Blues')

TP = np.diagonal(confMat)
FP = confMat.sum(axis=0) - TP
FN = confMat.sum(axis=1) - TP
TN = confMat.sum() - (TP + FP + FN)

def _report(TN, FP, FN, TP):
    TPR = TP/(TP+FN) if (TP+FN)!=0 else 0
    TNR = TN/(TN+FP) if (TN+FP)!=0 else 0
    PPV = TP/(TP+FP) if (TP+FP)!=0 else 0
    report = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 
              'TPR': TPR, 'Recall': TPR, 'Sensitivity': TPR,
              'TNR' : TNR, 'Specificity': TNR,
              'FPR': FP/(FP+TN) if (FP+TN)!=0 else 0,
              'FNR': FN/(FN+TP) if (FN+TP)!=0 else 0,
              'PPV': PPV, 'Precision': PPV,
              'F1 Score': 2*(PPV*TPR)/(PPV+TPR),
              'Accuracy': (TP+TN)/(TP+FP+FN+TN)
             }
    return report

def multi_classification_report(y_true, y_pred, labels=None, encoded_labels=True, as_frame=False):
    """
    Args:
        y_true (ndarray)
        y_pred (ndarray)
        labels (list)
        encoded_labels (bool): Need to be False if labels are not one hot encoded
        as_fram (bool): If True, return type will be DataFrame
        
    Return:
        report (dict)
    """
    
    conf_labels = None if encoded_labels else labels
    
    conf_mat = multilabel_confusion_matrix(y_test, y_pred, labels=conf_labels)
    report = dict()
    if labels == None:
        counter = np.arange(len(conf_mat))
    else:
        counter = labels
        
    for i, name in enumerate(counter):
        TN, FP, FN, TP = conf_mat[i].ravel()
        report[name] = _report(TN, FP, FN, TP)
    
    if as_frame:
        return pd.DataFrame(report)
    return report

def summarized_classification_report(y_true, y_pred, as_frame=False):
    """
    Args:
        y_true (ndarray)
        y_pred (ndarray)
        as_fram (bool): If True, return type will be DataFrame
        
    Return:
        report (dict)
    """

    report = dict()
    
    avg_list = ['micro', 'macro', 'weighted']
    for avg in avg_list:
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, average=avg)
        re = recall_score(y_true, y_pred, average=avg)
        f1 = f1_score(y_true, y_pred, average=avg)
        report[avg] = {'Accuracy': acc,
                       'Precision': pre,
                       'Recall': re,
                       'F1 Score': f1}
    if as_frame:
        return pd.DataFrame(report)
    return report

mcr = multi_classification_report(y_test, pred, labels=labels, encoded_labels=True, as_frame=True)
scr = summarized_classification_report(y_test, pred, as_frame=True)

# proposed model
mcr.to_csv(r'/Users/ahmad/Documents/Master/results/CICIoT2023/csvs/[CICIoT2023] [DNN] Binary classification report.csv')
scr.to_csv(r'/Users/ahmad/Documents/Master/results/CICIoT2023/csvs/[CICIoT2023] [DNN] Summarized binary classification report.csv')





















