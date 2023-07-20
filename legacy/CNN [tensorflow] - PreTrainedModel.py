from include.functions import initialise,load_data
from include.stats import plot_accuracy_loss
import sys
from keras.utils import np_utils
import sys
from keras import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from keras.models import load_model


def cnn_example():
 
    loadfromfile=True
    x_train,x_test,y_train,y_test,num_labels =load_data(load_from_file=loadfromfile,name= 'cnn')
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape

    x_train = x_train.reshape(x_train.shape[0], in_shape[1], in_shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[1], in_shape[0], 1)
    
    model = load_model('model//cnn_classifier.h5')

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    print(model.summary(), file=sys.stderr)
    #history = model.fit(x_train[:2], y_train[:2], batch_size=32, epochs=5,shuffle=True,validation_data=(x_test,y_test))
    #history = model.fit(x_train, y_train, batch_size=32, epochs=5,shuffle=True,validation_split=0.05)

    
    #result = model.predict(x_test)
    loss, acc = model.evaluate(x_test, y_test,verbose=1)
    #print("Accuracy: {:.2f}%".format(acc*100))
    #print("Loss: {:.2f}%".format(loss*100))
    #print("Highest Accuracy: {:.2f}%".format(max(history.history['val_accuracy'])*100))
    
    #plot_accuracy_loss(history=history)
    
    
    print('CNN Pre Trained')


if __name__ == "__main__":
    initialise()
    cnn_example()