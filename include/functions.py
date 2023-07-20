from . settings import _CLASS_LABELS,_DATA_PATH,mean_signal_length

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import joblib
import os
import sys
from typing import Tuple
#from scipy.io import wavfile
from sklearn.metrics import roc_curve, auc, classification_report,accuracy_score,confusion_matrix

import numpy as np
import scipy.io.wavfile as wav

from speechpy.feature import mfcc
import numpy as np
from sklearn.model_selection import train_test_split


"""
This example demonstrates how to use `CNN` model from
`speechemotionrecognition` package
"""
from keras.utils import np_utils
import matplotlib.pyplot as plt
import librosa
import soundfile
import tensorflow as tf


def load_data(load_from_file,name):
    if load_from_file:
        current_dir = os.getcwd()
        subdir = '\\training_data\\'
        save_dir = current_dir + subdir 
        
        x_train = joblib.load(os.path.join(save_dir, 'x_train_'+name+'.joblib'))
        x_test = joblib.load(os.path.join(save_dir, 'x_test_'+name+'.joblib'))
        y_train = joblib.load(os.path.join(save_dir, 'y_train_'+name+'.joblib'))
        y_test = joblib.load(os.path.join(save_dir, 'y_test_'+name+'.joblib'))
        num_labels = joblib.load(os.path.join(save_dir, 'num_labels_'+name+'.joblib'))
        #data = joblib.load(os.path.join(save_dir, 'data_'+name+'.joblib'))
        #labels = joblib.load(os.path.join(save_dir, 'labels_'+name+'.joblib'))
    else:
        x_train, x_test, y_train, y_test, num_labels = extract_data()
        current_dir = os.getcwd()
        subdir = '\\training_data\\'
        save_dir = current_dir + subdir 
        joblib.dump(x_train, os.path.join(save_dir, 'x_train_'+name+'.joblib'))
        joblib.dump(x_test, os.path.join(save_dir, 'x_test_'+name+'.joblib'))
        joblib.dump(y_train, os.path.join(save_dir, 'y_train_'+name+'.joblib'))
        joblib.dump(y_test, os.path.join(save_dir, 'y_test_'+name+'.joblib'))
        joblib.dump(num_labels, os.path.join(save_dir, 'num_labels_'+name+'.joblib')) 
        #joblib.dump(data, os.path.join(save_dir, 'data_'+name+'.joblib')) 
        #joblib.dump(labels, os.path.join(save_dir, 'labels_'+name+'.joblib')) 
    return x_train,x_test,y_train,y_test,num_labels#,data, labels


def initialise():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
    os.system('cls' if os.name == 'nt' else 'clear')
    """
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

    from keras import backend as K
    K.clear_session()

    from numba import cuda
    device = cuda.get_current_device()
    device.reset()
    """




def extract_data():
    data, labels = get_data(_DATA_PATH, class_labels=_CLASS_LABELS)

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(_CLASS_LABELS)#,data, labels

def get_data(data_path: str, flatten: bool = True, mfcc_len: int = 40,
             class_labels: Tuple = _CLASS_LABELS): 
             #-> \        Tuple[np.ndarray, np.ndarray]:

    data = []
    labels = []
    names = []
    cur_dir = os.getcwd()
    sys.stderr.write('curdir: %s\n' % cur_dir)
    data_path = _DATA_PATH
    ##data_path = 'D:\\PhD\\EmoDB'
    os.chdir(data_path)
    tempcount=0
    #count total files
    file_count = sum(len(files) for _, _, files in os.walk(data_path))    
    #count total files
    for i, directory in enumerate(class_labels):
        sys.stderr.write("started reading folder %s\n" % directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            tempcount+=1
            print("** "+str(tempcount)+" / "+str(file_count)+" **")            
            filepath = os.getcwd() + '/' + filename
            #feature_vector=extract_feature(filepath, mfcc=False, chroma=True, mel=False,contrast=False,tonnetz=False) 
            #************************************************************************************************************
            #************************************************************************************************************
            #************************************************************************************************************            
            feature_vector=extract_feature(filepath, mfcc=True, chroma=True, mel=True,contrast=True,tonnetz=True,num_features=40) 
            #************************************************************************************************************
            #************************************************************************************************************
            #************************************************************************************************************
            data.append(feature_vector)
            #print("****************************************")
            #print("Start:  " + str(feature_vector) + "End:  ")
            #print("****************************************")
            #print(str(feature_vector.shape)+"\n")
            #print("****************************************")
            labels.append(i)
            print("Procesing:  " + str(filename))
            #print("****************************************")
            names.append(filename)           
        sys.stderr.write("ended reading folder %s\n" % directory)
        os.chdir('..')
    os.chdir(cur_dir)
    return np.array(data), np.array(labels)


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    num_features = kwargs.get("num_features")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        ###################################
        s_len = len(X)
        if s_len < mean_signal_length:
            pad_len = mean_signal_length - s_len
            pad_rem = pad_len % 2
            pad_len //= 2
            X = np.pad(X, (pad_len, pad_len + pad_rem),
                            'constant', constant_values=0)
        else:
            pad_len = s_len - mean_signal_length
            pad_len //= 2
            X = X[pad_len:pad_len + mean_signal_length]
        ###################################
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.empty((num_features,0))
        if mfcc:
            mfcc = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_features))
            #print(str(mfcc.shape))
            result = np.hstack((result, mfcc))
            #print("MFCC Result:"+str(result) +"\n   Shape" + str(result.shape))
        if chroma:
            chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate,n_chroma=num_features))
            #print(str(chroma.shape))
            result = np.hstack((result, chroma))
            #print("CHROMA Result:"+str(result) +"\n  Shape" + str(result.shape))
        if mel:
            mel = np.array(librosa.feature.melspectrogram(X, sr=sample_rate,n_mels=num_features))
            result = np.hstack((result, mel))
            #print("MEL Result:"+str(result) +"\n  Shape" + str(result.shape))
        if contrast:
            contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate))
            result = np.hstack((result, np.pad(contrast,(0,num_features-7),'constant', constant_values=(0, 0))))
            #print("CONTRAST Result:"+str(result) +"\n  Shape" + str(result.shape))
        if tonnetz:
            tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate))
            result = np.hstack((result, np.pad(tonnetz,(0,num_features-6),'constant', constant_values=(0, 0))))
            #print("TONNETZ Result:"+str(result) +"\n  Shape" + str(result.shape))
    return result