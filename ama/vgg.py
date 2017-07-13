from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Dropout,Lambda
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization

class Vgg():

    def __init__(self, input_shape=(32,32,3)):
        self.model = None
        self.classes = None
        self.create(input_shape=input_shape)
    
    def ConvBlock(self, nb_filter, nb_layer):
        model = self.model
        for _ in range(nb_layer):
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(nb_filter,3,3,activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering='tf'))
                
    def FCBlock(self, nb_unit):
        model = self.model 
        model.add(Dense(nb_unit, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        
    def create(self, input_shape):
        model = self.model = Sequential()
        #Find mean of data set? and use a Lambda?
        #model.add(Lambda(lambda x: x/255, input_shape=input_shape,
        #    output_shape=input_shape))
        model.add(BatchNormalization(input_shape=input_shape))
        
        self.ConvBlock(32,2)
        self.ConvBlock(64,2)
        self.ConvBlock(128,2)
        self.ConvBlock(256,2)

        model.add(Flatten())

        self.FCBlock(512)
        
        model.add(Dense(17, activation='sigmoid'))
