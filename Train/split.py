#Import thu vien
import wx
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA
import joblib

#Nhap file du lieu dau vao
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_info_rows',None)
pd.set_option('display.max_columns', None)
filename=input()
result = pd.read_csv(filename, delimiter=";")
data=pd.DataFrame(result)
print(data.head(5))

#Phan thanh train test theo ty le lua chon va luu vao csv
X, y=train_test_split(data, test_size=0.20)
print(X.shape)
print(y.shape)
print(X.head(5))
print(y.head(5))
X.to_csv('train.csv')
y.to_csv('test.csv')