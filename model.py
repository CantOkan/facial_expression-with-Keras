import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation,Flatten,Dropout,Dense


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

num_classes=7
img_row,img_col=48,48

batch_size=16

train_data_dir="/home/canok/Desktop/TransferLearning/EmotionDetector/fer2013/train"

val_data_dir="/home/canok/Desktop/TransferLearning/EmotionDetector/fer2013/validation"



tran_datagen=ImageDataGenerator(

    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode="nearest"
)


val_datagen=ImageDataGenerator(rescale=1./255)

train_generator=tran_datagen.flow_from_directory(
    train_data_dir,
    color_mode="grayscale",
    target_size=(img_row,img_col),batch_size=batch_size
)

val_generator=val_datagen.flow_from_directory(
    val_data_dir,
    color_mode="grayscale",
    target_size=(img_row,img_col),batch_size=batch_size
)




model=Sequential()

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer="he_normal",input_shape=(img_row,img_col,1)))
model.add(Activation("elu"))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer="he_normal"))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

#Block #2:second Conv
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer="he_normal"))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer="he_normal"))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))


# Block #3: third CONV => RELU => CONV => RELU => POOL
# layer set
model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# Block #4: third CONV => RELU => CONV => RELU => POOL
# layer set
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# Block #5: first set of FC => RELU layers
model.add(Flatten())
model.add(Dense(64, kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# Block #6: second set of FC => RELU layers
model.add(Dense(64, kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block #7: softmax classifier
model.add(Dense(num_classes, kernel_initializer="he_normal"))
model.add(Activation("softmax"))



checkpoint = ModelCheckpoint("/home/canok/Desktop/TransferLearning/EmotionDetector/emotion_little_vgg_3.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)


earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)


reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)

callbacks = [earlystop, checkpoint, reduce_lr]

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics = ['accuracy'])


nb_train_samples = 28273
nb_validation_samples = 3534
epochs = 10


history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=nb_validation_samples//batch_size
)

model.save('my_model.h5')