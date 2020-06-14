from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


img_rows, img_cols = 128,128 # Dimension of the image
VGG = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols,3))
# include_top=False is excluding he top layer of VGG

for layers in VGG.layers:
    layers.trainable = False 
# Since we are going to train the last of the VGG16 so that we are ignoring the first and training only the last layyer

# Defining the last layer of VGG
def addTopModelVGG(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(512,activation='relu')(top_model) # 512 neurons and relu activation
    top_model = Dense(256,activation='relu')(top_model) # 256 neurons and relu activation
    top_model = Dense(256,activation='relu')(top_model) # 256 neurons and relu activation
    top_model = Dense(128,activation='relu')(top_model) # 128 neurons and relu activation
    top_model = Dense(128,activation='relu')(top_model) # 128 neurons and relu activation
    top_model = Dense(num_classes,activation='softmax')(top_model) # softmax activation
    return top_model
 
 num_classes = 6
FC_Head = addTopModelVGG(VGG, num_classes) # Calling addTopModelVGG function
# Parameters indicate the bottom model and the number of classes
model = Model(inputs = VGG.input, output = FC_Head)
print(model.summary())

# Training data directory
train_data_dir = 'train' 
# Validation data directory
validation_data_dir = 'validation'

# Genertation more images from a single image
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 30,
                                   width_shift_range = 0.3,
                                   height_shift_range = 0.3,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')

# For Cross checking the training data
validation_datagen = ImageDataGenerator(rescale = 1./255)

# Batch size indicates the number of images to be traimed at a time
batch_size = 32

# More features to be added to the training data
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                   target_size = (img_rows, img_cols),
                                                   batch_size = batch_size,
                                                   class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                             target_size = (img_rows, img_cols),
                                                             batch_size = batch_size,
                                                             class_mode = 'categorical')

from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ModelCheckpoint - Save only those with the best accuracy
checkpoint = ModelCheckpoint('finger_sign.h5',
                            monitor = 'val_loss', # It monitors loss or not
                            mode = 'min',
                            save_best_only = True, # Save only the best 
                            verbose = 1)
                        
# EarlyStopping - If model validation is not improving then we stop
earlystop = EarlyStopping(monitor = 'val_loss',
                         min_delta = 0,
                         restore_best_weights = True,
                         patience = 10, # If patience = 3 means that the model validation doesn't increase for 10 rounds it stop
                         verbose = 1)

# ReduceLROnPlateau - Reduce learning rate. If the model accuracy is not improving reduce the learning rate
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss',
                                            patience = 5, # If patience = 3 means that the model validation doesn't increase for 5 rounds it reduce the learning rate
                                             verbose = 1,
                                             factor = 0.2,
                                             min_lr = 0.0001)

callbacks = [earlystop, checkpoint, learning_rate_reduction]

model.compile(loss = 'categorical_crossentropy',
             optimizer = Adam(lr = 0.001),
             metrics = ['accuracy'])

nb_train_samples = 1200
nb_validaton_samples = 300

epochs = 10 # Number of times want to be trained
batch_size = 32 # number of images to be traimed at a time

history = model.fit_generator(train_generator,
                             steps_per_epoch = nb_train_samples//batch_size,
                             epochs = epochs,
                             callbacks = callbacks,
                             validation_data = validation_generator,
                             validation_steps = nb_validaton_samples//batch_size)