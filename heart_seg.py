from keras.models import Sequential
from keras.models import Dropout, Activation
from keras.models import Convolutioanl2D, MaxPooling2D

model = Sequential()
model.add(Convolutioanl2D(32, 3, 3, border_mode='valid', input_shape=(1, 224, 224)))
model.add(Activation('relu'))
model.add(Convolutioanl2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolutioanl2D(512, 1, 1, init='uniform', activation='relu', border_mode='valid'))

model.add(Convolutioanl2D(512, 1, 1, init='uniform', activation='relu', border_mode='valid'))

model.add(Convolutioanl2D(1, 1, 1, init='uniform', activation='relu', border_mode='valid'))
#output image = 56x56

#upsample the output to match input dimensions = 224x224