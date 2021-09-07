# importing libraries
import numpy as np 
import pandas as pd 
import cv2
import PIL
from PIL import Image
from numpy import asarray
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import (Activation,Dropout,Flatten,Dense,Input,Conv2D,MaxPool2D)
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras import backend as K
from keras import metrics, applications, optimizers, models,layers
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import skimage.segmentation as seg
import skimage.filters as filters
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from keras import applications
from keras.models import Model
from keras.optimizers import SGD, Adam
from sklearn.metrics import roc_auc_score
from keras.layers import GlobalAveragePooling2D
%matplotlib inline
import matplotlib.image as mpimg
from IPython.display import Image, display
from keras.applications.vgg16 import VGG16,preprocess_input
import plotly.graph_objs as go
import plotly.graph_objects as go
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model,load_model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc
import skimage.io
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
#from livelossplot import PlotLossesKeras
#pip install livelossplot
#import sys
#sys.path.insert(0, '/Users/mikko/Documents/GitHub/talos')
#import talos

import os

# establishing datasets
train = pd.read_csv('../input/melanoma-external-malignant-256/train_concat.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

train['patient_id']
train['sex'] = train['sex'].map({'male':0,'female':1})
test['sex'] = test['sex'].map({'male':0,'female':1})
train['sex'] = train['sex'].fillna(0.5)
test['sex'] = test['sex'].fillna(0.5)
train['image_name_data'] = train['image_name'].apply(lambda x: x[5:12])
test['image_name_data'] = test['image_name'].apply(lambda x: x[5:12])
train['patient_id']  = train['patient_id'].fillna('9999999')
test['patient_id'] = test['patient_id'].fillna('9999999')
train['patient_id_data'] = train['patient_id'].apply(lambda x: x[-7:])
test['patient_id_data'] = test['patient_id'].apply(lambda x: x[-7:])
train['patient_id_data']= train['patient_id_data'].astype(int)
test['patient_id_data'] = test['patient_id_data'].astype(int)
train['age_null'] = train['age_approx'].apply(lambda x:pd.isnull(x))
train['age_approx'] = train['age_approx'].fillna(49)
test['age_null'] = test['age_approx'].apply(lambda x: pd.isnull(x))
test['age_approx'] = test['age_approx'].fillna(49)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].map({'head/neck':1,'upper extremity':1,'lower extremity':2,'torso':3,'palms/soles':4,'oral/genital':5,
                                                                                    'anterior torso':3,'lateral torso':3,'posterior torso':3})
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].map({'head/neck':1,'upper extremity':1,'lower extremity':2,'torso':3,'palms/soles':4,'oral/genital':5})
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna(0)
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna(0)
train = pd.concat([train,pd.get_dummies(train['anatom_site_general_challenge'],dummy_na = True, prefix = 'anatom_site_general_challenge')],axis = 1)
test = pd.concat([test,pd.get_dummies(test['anatom_site_general_challenge'],dummy_na = True, prefix = 'anatom_site_general_challenge')],axis = 1)
train['image_path'] = train['image_name'].apply(lambda x: '../input/melanoma-external-malignant-256/train/train/' + x + '.jpg')
test['image_path'] = test['image_name'].apply(lambda x: '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/' + x + '.jpg')
train = train.drop(columns = ['image_name','patient_id'])
test = test.drop(columns = ['image_name','patient_id'])
train['anatom_site_general_challenge_0.0'] = 0

train_df = pd.DataFrame()

count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
train_image_dir = '../input/siim-isic-melanoma-classification/jpeg/train/'
for i in range (0, len(train)):
    if train['anatom_site_general_challenge'][i] == 1 and train['target'][i] == 1:
        train_df = train_df.append(train[i:i+1])
        count1+=1
    elif train['anatom_site_general_challenge'][i] == 2 and train['target'][i] == 1: 
        train_df = train_df.append(train[i:i+1])
        count2 +=1
    elif train['anatom_site_general_challenge'][i] == 3 and train['target'][i] == 1:
        train_df = train_df.append(train[i:i+1])
        count3 +=1
    elif train['anatom_site_general_challenge'][i] == 4 and train['target'][i] == 1:
        train_df = train_df.append(train[i:i+1])
        count4+=1
    elif train['anatom_site_general_challenge'][i] == 5 and train['target'][i] == 1: 
        train_df = train_df.append(train[i:i+1])
        count5+=1
    elif train['anatom_site_general_challenge'][i] == 6 and train['target'][i] == 1:
        train_df = train_df.append(train[i:i+1])
        count6 +=1
    elif train['anatom_site_general_challenge'][i] == 0 and train['target'][i] == 1:
        train_df = train_df.append(train[i:i+1])
        count7+=1
countt1=0
countt2=0
countt3=0
countt4=0
countt5=0
countt6=0
countt7=0
for i in range (0,len(train)):
    if train['anatom_site_general_challenge'][i] == 1 and train['target'][i] == 0 and countt1<count1:
        train_df = train_df.append(train[i:i+1])
        countt1+=1
    elif train['anatom_site_general_challenge'][i] == 2 and train['target'][i] == 0 and countt2 < count2:
        train_df = train_df.append(train[i:i+1])
        countt2+=1
    elif train['anatom_site_general_challenge'][i] == 3 and train['target'][i] == 0 and countt3 <count3:
        train_df = train_df.append(train[i:i+1])
        countt3 +=1
    elif train['anatom_site_general_challenge'][i] == 4 and train['target'][i] == 0 and countt4 <count4:
        train_df = train_df.append(train[i:i+1])
        countt4 +=1
    elif train['anatom_site_general_challenge'][i] == 5 and train['target'][i] == 0 and countt5 < count5: 
        train_df = train_df.append(train[i:i+1])
        countt5 += 1
    elif train['anatom_site_general_challenge'][i] == 6 and train['target'][i] == 0 and countt6 < count6:
        train_df = train_df.append(train[i:i+1])
        countt6 += 1 
    elif train['target'][i] == 0 and countt7 < count7:
        train_df = train_df.append(train[i:i+1])
        countt7 += 1
     
# functions
def display_one (a,title1="Original"):
    plt.imshow(a),plt.title(title1)
    plt.xticks([]),plt.yticks([])
    plt.show()
def display(a, b, title1="Original",title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]),plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]),plt.yticks([])
    plt.show()
nrows = 150
ncolumns = 150
channels = 3
import timeit
def process (images):
    x = []
    i=0
    for image in images:
        start = timeit.default_timer()
        x.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncolumns),interpolation = cv2.INTER_CUBIC))
        print(i,'Time: ',timeit.default_timer()-start)
        i+=1
        
    return x;
def no_noise(train_df):
    no_noise_data = []
    for i in range(len(train_df)):
        blur = cv2.GaussianBlur(train_df[i],(5,5),0)
        no_noise_data.append(blur)
    return no_noise_data
def eroded (train_df):
    kernel = np.ones((3,3),np.uint8)
    eroded=[]
    no_noise_data = no_noise(train_df)
    for i in no_noise_data:
        dil = cv2.erode(i,kernel,iterations = 1)
        #dil = cv2.cvtColor(dil,cv2.COLOR_BGR2GRAY)
        eroded.append(dil)
    return np.array(eroded)
kernel_sharpening = np.array([[-1,-1,-1],
                             [-1,9,-1],
                             [-1,-1,-1]])
def preprocess(images):
    preprocessed = []
    for img in images:
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/cdf.max()

        cdf_m = np.ma.masked_equal (cdf,0)
        cdf_m = (cdf_m - cdf_m.min()) * 255/(cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')

        img2 = cdf[img]
        img3 = cv2.GaussianBlur(img2,(5,5),0)

        img4 = cv2.filter2D(img3,-1,kernel_sharpening)
        preprocessed.append(img4)
    preprocessed = np.array(preprocessed)
    return preprocessed
def plot():
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Accuracy')
    plt.ylabel = ('AUC')
    plt.xlabel = ('epoch')
    plt.legend(['train','test'],loc = 'upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upper left')
    plt.show()
datagen = ImageDataGenerator (horizontal_flip = True, vertical_flip = True, zoom_range = [0.5,1.0],brightness_range = [0.2,1.0])
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy

#InceptionResNetV2
image_data = process(train_df['image_path'])
image_data = preprocess(image_data)
#image_data.size
#len(image_data)
#image_data.size/10212/3/150
y = train_df['target']
train_x,val_x,train_y,val_y = train_test_split (image_data,y, test_size = 0.2, random_state = 262)
#datagen = ImageDataGenerator (horizontal_flip = True, vertical_flip = True, zoom_range = [0.5,1.0],brightness_range = [0.2,1.0],validation_split=0.2)
#datagen.fit(train_x)
incRes = applications.InceptionResNetV2(include_top = True,weights = 'imagenet',input_shape=(299,299,3))
output = incRes.output
#output = Dropout(0.9)(output)
predictions = Dense(1,activation = 'sigmoid')(output)
incRes = Model(inputs = incRes.input, outputs = predictions)
epochs = 100 #5757
learning_rate = 0.0001
decay_rate = learning_rate/ epochs
sgd = SGD (lr = learning_rate,momentum = 0.8,decay = decay_rate,nesterov = False)
adam = Adam (lr = 1e-5)
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping (monitor = 'val_auc_1',mode='max',verbose = 1,patience = 5)
incRes.compile(optimizer = adam, loss='binary_crossentropy',metrics = [metrics.AUC()])
history = incRes.fit(train_x,train_y,validation_split = 0.2, epochs = 25, batch_size = 40, verbose = 1, callbacks = [es])
#plot()
predictions = incRes.predict(val_x)
print("InceptionResNetV2: ", roc_auc_score(val_y,predictions))

#submission
submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
data = process(test['image_path'])
data = eroded(data)
preds = incRes.predict(data)
submission['target']=preds
submission.to_csv('submission.csv',index = False)
