from include.functions import initialise, load_data
from include.stats import plot_accuracy_loss
from keras.utils import np_utils
import sys
from keras import Sequential
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, MaxPooling1D, Layer

import tensorflow as tf

class DendriticNeuronLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_dendrites):
        super(DendriticNeuronLayer, self).__init__()
        self.units = units
        self.num_dendrites = num_dendrites

    def build(self, input_shape):
        # Create a kernel weight for each dendrite
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units, self.num_dendrites),
            initializer='uniform',
            trainable=True)

    def call(self, inputs):
        dendritic_outputs = tf.tensordot(inputs, self.kernel, axes=1)
        soma_input = tf.reduce_sum(dendritic_outputs, axis=-1)
        return tf.keras.activations.sigmoid(soma_input)




def conv_dnm_example():

    #loadfromfile=False
    loadfromfile=True
    x_train,x_test,y_train,y_test,num_labels =load_data(load_from_file=loadfromfile,name= 'conv_dnm')

       
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    num_timesteps, num_features = x_train[0].shape[0], x_train[0].shape[1]
    num_classes = num_labels


    print("num_timesteps:  " + str(num_timesteps))
    print("num_features:  " + str(num_features))
    print("num_classes:  "+ str(num_classes))

    #####################
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(num_timesteps, num_features)))
    DendriticNeuronLayer(64, num_dendrites=5)
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    """    
    model = Sequential()
    model.add(Conv1D(128, kernel_size=5, activation='relu', input_shape=(num_timesteps, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=True))
    DendriticNeuronLayer(64, num_dendrites=5)
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    


    history = model.fit(x_train, y_train, batch_size=32, epochs=50,shuffle=True,validation_data=(x_test,y_test))

    loss, acc = model.evaluate(x_test, y_test)
    print("Accuracy: {:.2f}%".format(acc*100))
    print("Loss: {:.2f}%".format(loss*100))
    print("Highest Accuracy: {:.2f}%".format(max(history.history['val_accuracy'])*100))
    
    plot_accuracy_loss(history=history)
    
    model.save("model/conv_dnm_classifier.h5") 
    
    print('conv_dnm Done')
    

if __name__ == "__main__":
    initialise()
    conv_dnm_example()