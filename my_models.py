import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.initializers import random_uniform, glorot_uniform

def identity_block(X, f,filters, training = True, initializer=random_uniform):
    F1,F2,F3=filters
    X_short = X
    
    X = tfl.Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training = training) # Default axis
    X = tfl.Activation('relu')(X)

    X = tfl.Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training = training) # Default axis
    X = tfl.Activation('relu')(X)

    X = tfl.Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training = training) # Default axis

    X = tfl.Add()([X_short,X])
    X = tfl.Activation('relu')(X)
    return X

def convolutional_block(X, f, filters, s = 2, training=True, initializer=random_uniform):
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    
    # First component of main path glorot_uniform(seed=0)
    X = tfl.Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training=training)
    X = tfl.Activation('relu')(X)

    ### START CODE HERE
    
    ## Second component of main path (≈3 lines)
    X = tfl.Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training = training) # Default axis
    X = tfl.Activation('relu')(X)

    ## Third component of main path (≈2 lines)
    X = tfl.Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training = training) # Default axis
    
    
    ##### SHORTCUT PATH ##### (≈2 lines)
    X_shortcut = tfl.Conv2D(filters = F3, kernel_size = (1,1),strides = (s,s),padding = 'valid', kernel_initializer = initializer(seed=0))(X_shortcut)
    X_shortcut = tfl.BatchNormalization()(X_shortcut, training = training)
    
    ### END CODE HERE

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = tfl.Add()([X, X_shortcut])
    X = tfl.Activation('relu')(X)
    return X

def ResNet50(input_shape=(180,180,3),classes = 5):
    inputs = tf.keras.Input(input_shape)
    X = tfl.Rescaling(1./255)(inputs)
    X = tfl.ZeroPadding2D((3,3))(X)

    # Stage 1
    X = tfl.Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X)
    X = tfl.Activation('relu')(X)
    X = tfl.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    ### START CODE HERE
    
    ## Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3,filters = [128,128,512] ,s=2)
    X = identity_block(X, f=3, filters = [128,128,512])
    X = identity_block(X, f=3, filters = [128,128,512])
    X = identity_block(X, f=3, filters = [128,128,512])
    
    ## Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3,filters = [256,256,1024] ,s=2)
    X = identity_block(X, f=3, filters = [256,256,1024])
    X = identity_block(X, f=3, filters = [256,256,1024])
    X = identity_block(X, f=3, filters = [256,256,1024])
    X = identity_block(X, f=3, filters = [256,256,1024])
    X = identity_block(X, f=3, filters = [256,256,1024])

    ## Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3,filters = [512,512,2048] ,s=2)
    X = identity_block(X, f=3, filters = [512,512,2048])
    X = identity_block(X, f=3, filters = [512,512,2048])

    ## AVGPOOL (≈1 line). Use(X) "X = AveragePooling2D(...)(X)"
    X = tfl.AveragePooling2D((2,2))(X)
    
    ### END CODE HERE

    # output layer
    X = tfl.Flatten()(X)
    X = tfl.Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = tf.keras.Model(inputs = inputs, outputs = X)

    return model

if __name__=='__main__':
    model = ResNet50()
    model.summary()