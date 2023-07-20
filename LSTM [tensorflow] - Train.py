from include.functions import initialise,load_data
from include.stats import plot_accuracy_loss
import sys
from keras.utils import np_utils
import sys
from keras import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout


def lstm_example():
 
    #loadfromfile=False
    loadfromfile=True
    
    x_train,x_test,y_train,y_test,num_labels =load_data(load_from_file=loadfromfile,name= 'lstm')

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    num_timesteps, num_features = x_train[0].shape[0], x_train[0].shape[1]
    num_classes = num_labels


    print("num_timesteps:  " + str(num_timesteps))
    print("num_features:  " + str(num_features))
    print("num_classes:  "+ str(num_classes))

    #####################
    model = Sequential()
    model.add(LSTM(128, input_shape=(num_timesteps, num_features)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(num_labels, activation='softmax'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    print(model.summary(), file=sys.stderr)
    

    history = model.fit(x_train, y_train, batch_size=32, epochs=50,shuffle=True,validation_data=(x_test,y_test))

    loss, acc = model.evaluate(x_test, y_test)
    print("Accuracy: {:.2f}%".format(acc*100))
    print("Loss: {:.2f}%".format(loss*100))
    print("Highest Accuracy: {:.2f}%".format(max(history.history['val_accuracy'])*100))
    
    plot_accuracy_loss(history=history)
    

    model.save("model/lstm_classifier.h5") 
    #model = tf.keras.models.load_model('path/to/location')
    print('LSTM Done')





if __name__ == "__main__":
    initialise()
    #lstm_example()
    lstm_example()