# Need to design a classifier which can tell if Apple's Logo is present in the given image or NOT.
# Using 256x256 dimension square grayscaled images as the object has very specific structure and doesn't depend on Color.


from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing import image


import numpy as np
import matplotlib as plt

# Input 
# Training and testing data is present in differnt directories. Both containing positive and negetive examples.
# Images present in the directory aren't labelled but their names can be used as a label.
# Names of Negative example starts with an uppercase letter.

# First load the training data (images(greyscale) in a numpy array).
# Load the corresponding label using the filename of the images. It will work as the read order is preserved.
# Labels are going to an array of two elements (0/1) 0 indicates absense and 1 indicates presence of Apple's logo.

train_datagen = image.ImageDataGenerator(rescale = 1./255,
			     rotation_range = 45,
			     width_shift_range = 0.2,
			     height_shift_range = 0.2,
			     shear_range = 0.2,
			     zoom_range = 0.2,
			    )

train_generator = train_datagen.flow_from_directory('./train/',
			      target_size = (128,128),
			      color_mode = 'grayscale',
			      class_mode = 'categorical',
			      )

validation_datagen = image.ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory('./validation/',
				target_size = (128,128),
				class_mode = 'categorical',
				color_mode = 'grayscale',
				) 


# Architechture of the CNN on the lines of VGG16 but not very deep.
model = Sequential()

model.add( Conv2D(32, (5, 5), activation='relu', input_shape=(128,128,1)) )
model.add( Conv2D(32, (5, 5), activation='relu',  ))
model.add( MaxPooling2D(pool_size = (2,2)) )

model.add( Conv2D(64, (3, 3), activation='relu', ))
model.add( Conv2D(64, (3, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size = (2,2)) )

model.add( Conv2D(96,(3, 3), activation='relu', ))
model.add( Conv2D(96,(3, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size = (2,2)) )

# Flatten the 3D data to a single dimensional numpy array.
model.add(Flatten())

# Fully connected Dense layer on Flattened image
model.add( Dense(128, activation='relu'))
model.add( Dense(64 , activation='relu'))
model.add( Dense(2, activation='softmax' ))

# Specify the optimizer and loss functions to be used.
# For fine grained control on the optimizer, an instance of it is created.
epochs = 25
lrate = 0.01
decay = lrate/epochs

sgd = SGD( lr=lrate, momentum=0.9 , decay=decay )

# Loss function binary_crossentropy should be prefered.(only two classes)

model.compile(loss = 'categorical_crossentropy',
	      optimizer = 'SGD',
	      metrics= ['accuracy'],
              )

print(model.summary())
model.fit_generator(train_generator,
		    epochs=5,
		    validation_data = validation_generator,

		    )
model.save_weights('tI128x256x128.h5')

#score = model.evaluate()
#print("Accuracy: %.2f%%" % (scores[1]*100))



