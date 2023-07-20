from include.functions import initialise,load_data,extract_feature,get_data
from include.settings import _CLASS_LABELS,_DATA_PATH,mean_signal_length

from include.stats import plot_accuracy_loss
import sys
from keras.utils import np_utils
import sys
from keras import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from keras.models import load_model
import tensorflow as tf

import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split

import random
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '3'


def clear_terminal():
    import os
    os.system('clear')

def choose_random_file(root_path):
    files = []
    # os.walk returns a generator that creates a tuple of values
    # for each directory in the tree rooted at the directory given in the path.
    for root, dirs, filenames in os.walk(root_path):
        for filename in filenames:
            # join the root directory path and the filename
            file_path = os.path.join(root, filename)
            files.append(file_path)
    return random.choice(files)

def clstm_classify(count):
 
    
    #model = load_model('model//conv_lstm_classifier.h5')
    #model = load_model('model//conv_lstm_transfer_classifier.h5')
    model = load_model('model//conv_lstm_classifier-backup.h5')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #random_num=random.randint(1,698)
    emotion_list = ["Anger", "Anticipation","Disgust","Fear","Joy","Sadness","Suprise","Trust"]

    random_file = choose_random_file("C:\\Phd\\Combined_Dataset_Train_Transfer\\Anger")
    #random_file = choose_random_file("C:\\Phd\\Afrikaans\\Anticipation")

    
    print("Iteration - "+ str(count) +"     Selected File: "+random_file)
    features = extract_feature(random_file, mfcc=True, chroma=True, mel=True,contrast=True,tonnetz=True,num_features=40) 
    results = model.predict(features.reshape(1,40, 1047))

    result_string = str(results[0])
    result_string = result_string.replace('[','')
    result_string = result_string.replace(']','')
    result_string = result_string.split(' ')

    sum=0.00

    #print("***********NORMAL*********************\n")
    for x in range(len(emotion_list)):
        #str_num = result_string[x]
        try:
            num = float(result_string[x])*100
            sum+=num
        except ValueError:
            print("Cannot convert string to float")
        str_num = f"{num:.2f}"
        #print("Emotion: " + emotion_list[x] + ":  "+ str(str_num)+"%" )
        print(str(str_num), end=' ')

    #print("Total: "+str(f"{sum:.2f}"))    
    print("\n")
    



if __name__ == "__main__":
    initialise()
    clear_terminal()
    
    for i in range(1,10):
      clstm_classify(i)