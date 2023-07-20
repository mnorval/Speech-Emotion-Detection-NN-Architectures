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



def lstm_classify():
 
    loadfromfile=True
    x_train,x_test,y_train,y_test,num_labels =load_data(load_from_file=loadfromfile,name= 'lstm')

    model = load_model('model//lstm_classifier.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
    #print(model.summary(), file=sys.stderr)
    #filename="C:\\Phd\\Afrikaans\\Anger\\[7de Laan][Amorey Welman][Anger]-1.wav"
    #features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(40,1047)
    

    
    
    random_num=random.randint(1,698)
    emotion_list = ["Anger", "Anticipation","Disgust","Fear","Joy","Sadness","Suprise","Trust"]
    emotion = int(y_train[random_num])

    
    print("Number selected: " + str(random_num) + "  Emotion:  "+ emotion_list[emotion] + "\n\n")

    filename="C:\\Phd\\Afrikaans\\Anger\\[7de Laan][Amorey Welman][Anger]-1.wav"
    #features = extract_feature(filename, mfcc=True, chroma=True, mel=True)
    #features = get_data(filename,  _CLASS_LABELS)
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True,contrast=True,tonnetz=True,num_features=40)  
    
    results = model.predict(features.reshape(1,40, 1047))

    
    #print(results[0:4])
    result_string = str(results[0])
    result_string = result_string.replace('[','')
    result_string = result_string.replace(']','')
    result_string = result_string.split(' ')

    print("***********NORMAL*********************\n")
    for x in range(len(emotion_list)):
        temp_result = round(float(result_string[x]),4)*100
        print("Emotion: " + emotion_list[x] + "  Result: "+ str(temp_result)+"%" )



    
    #print('LSTM Done')




if __name__ == "__main__":
    initialise()
    lstm_classify()