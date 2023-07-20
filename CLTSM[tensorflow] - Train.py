from include.functions import initialise, load_data
from include.stats import plot_accuracy_loss
from keras.utils import np_utils
import sys
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, MaxPooling1D

def conv_lstm_example():

    #loadfromfile=False
    loadfromfile=True
    x_train,x_test,y_train,y_test,num_labels =load_data(load_from_file=loadfromfile,name= 'conv_lstm')

       
    print("Before Reshape")
    print("x_train.shape :" +  str(x_train.shape) +"    y_train.shape:"+ str(y_train.shape))
    print("x_test.shape :" +  str(x_test.shape) +"    y_test.shape:"+ str(y_test.shape))

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape

    
    #x_train = x_train.reshape(x_train.shape[0], in_shape[1], in_shape[0], 1)
    #x_test = x_test.reshape(x_test.shape[0], in_shape[1], in_shape[0], 1)

    print("After Reshape")
    print("x_train.shape :" +  str(x_train.shape) +"    y_train.shape:"+ str(y_train.shape))
    print("x_test.shape :" +  str(x_test.shape) +"    y_test.shape:"+ str(y_test.shape))
    

    num_timesteps, num_features = x_train.shape[1], x_train.shape[2]
    num_classes = y_train.shape[1]
    
   

    print("num_timesteps, num_features, num_classes")
    print("num_timesteps:" +str(num_timesteps) )
    print("num_features:" +str(num_features))
    print("num_classes:" +str(num_classes))

    #####################
   
    model = Sequential()
    model.add(Conv1D(128, kernel_size=5, activation='relu', input_shape=(num_timesteps, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
    
  
    #####################
    
    history = model.fit(x_train, y_train,  batch_size=32, epochs=20,shuffle=True,validation_data=(x_test,y_test))
    #history = model.fit(x_train, y_train, batch_size=32, epochs=50,shuffle=True,validation_data=(x_test,y_test))
    #history = model.fit(x_train, y_train, batch_size=32, epochs=5,shuffle=True,validation_split=0.05)


    loss, acc = model.evaluate(x_test, y_test)
    print("Accuracy: {:.2f}%".format(acc*100))
    print("Loss: {:.2f}%".format(loss*100))
    print("Highest Accuracy: {:.2f}%".format(max(history.history['val_accuracy'])*100))
    
    plot_accuracy_loss(history=history)

    
    model.save("model/conv_lstm_classifier.h5") 
   
    print('conv_lstm Done')
    

if __name__ == "__main__":
    initialise()
    conv_lstm_example()