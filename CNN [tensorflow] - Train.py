from include.functions import initialise, load_data
from include.stats import plot_accuracy_loss
from keras.utils import np_utils
import sys
from keras import Sequential
from keras.layers import LSTM,Dense, Dropout, Conv2D, Flatten,BatchNormalization, Activation, MaxPooling2D,Reshape



def cnn_example():

    #loadfromfile=False
    loadfromfile=True
    x_train,x_test,y_train,y_test,num_labels =load_data(load_from_file=loadfromfile,name= 'cnn')

       
 
    

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape

    x_train = x_train.reshape(x_train.shape[0], in_shape[1], in_shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[1], in_shape[0], 1)

    print("x_train.shape :" +  str(x_train.shape))
    print("x_test.shape :" +  str(x_test.shape))
    #####################
    
    model = Sequential()
    model.add(Conv2D(8, (13, 13), input_shape=(in_shape[1], in_shape[0], 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (13, 13)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(8, (13, 13)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_labels, activation='softmax')) 
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    print(model.summary(), file=sys.stderr)
    
 

    history = model.fit(x_train[:2], y_train[:2], batch_size=32, epochs=50,shuffle=True,validation_data=(x_test,y_test))
    #history = model.fit(x_train, y_train, batch_size=32, epochs=5,shuffle=True,validation_split=0.05)


    loss, acc = model.evaluate(x_test, y_test)
    print("Accuracy: {:.2f}%".format(acc*100))
    print("Loss: {:.2f}%".format(loss*100))
    print("Highest Accuracy: {:.2f}%".format(max(history.history['val_accuracy'])*100))
    
    plot_accuracy_loss(history=history)

    
    model.save("model/cnn_classifier.h5") 
   
    print('CNN Done')


if __name__ == "__main__":
    initialise()
    cnn_example()