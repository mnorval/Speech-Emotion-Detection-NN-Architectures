# import the module
import soundfile as sf
import os
import sys
import numpy as np
from statistics import mean

_DATA_PATH = "C:\PhD\Combined_Dataset"
_CLASS_LABELS =  ("Anger", "Anticipation","Disgust","Fear","Joy","Sadness","Suprise","Trust")
#_DATA_PATH = "D:\\Phd\\EmoDB"
# _CLASS_LABELS =  ("Anger", "Anxiety","Boredom","Disgust","Happiness","Neutral","Sadness")

data = []

cur_dir = os.getcwd()
sys.stderr.write('curdir: %s\n' % cur_dir)
os.chdir(_DATA_PATH)
for i, directory in enumerate(_CLASS_LABELS):
    sys.stderr.write("started reading folder %s\n" % directory)
    os.chdir(directory)
    for filename in os.listdir('.'):
        filepath = os.getcwd() + '/' + filename
        data_, samplerate_ = sf.read(filepath)        
        print("Length: "+filename+":  ", str(len(data_)))
        data.append(len(data_))
    sys.stderr.write("ended reading folder %s\n" % directory)
    os.chdir('..')
os.chdir(cur_dir)
print("Max: "+str(max(data)))
print("Min: " + str(min(data)))
print("Average: " + str(mean(data)))
