from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 224, 224)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(512, 1, 1, init='uniform', activation='relu', border_mode='valid'))

model.add(Convolution2D(512, 1, 1, init='uniform', activation='relu', border_mode='valid'))

model.add(Convolution2D(1, 1, 1, init='uniform', activation='relu', border_mode='valid'))
#output image = 56x56

#upsample the output to match input dimensions = 224x224
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=1, batch_size=32)