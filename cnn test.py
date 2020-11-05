# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 00:26:28 2020

@author: Henock
"""



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
 


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

batch_size = 5





train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Testing Augmentation - Only Rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)

# Generates batches of Augmented Image data
train_generator = train_datagen.flow_from_directory('TrainTestData/Train', target_size = (300, 300), 
                                                    batch_size = batch_size,
                                                    class_mode = 'binary') 

# Generator for validation data
validation_generator = test_datagen.flow_from_directory('TrainTestData/Test', 
                                                        target_size = (300, 300),
                                                        batch_size = batch_size,
                                                        class_mode = 'binary')



# Fit the model on Training data
history = model.fit_generator(train_generator,
                    epochs = 50,
                    validation_data = validation_generator,
                    verbose = 1)

# Evaluating model performance on Testing data
loss, accuracy = model.evaluate(validation_generator)

print("\nModel's Evaluation Metrics: ")
print("---------------------------")
print("Accuracy: {} \nLoss: {}".format(accuracy, loss))



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# model.save_weights("top_model_weights_path")
# model.save("topmodel")


temp=np.argmax(model.predict_generator(validation_generator), axis=1)



test_steps_per_epoch = np.math.ceil(validation_generator.samples / validation_generator.batch_size)
predictions = model.predict_generator(validation_generator, steps=test_steps_per_epoch)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)


true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())  

report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)  




# # all normal Test images

# import glob
# from PIL import Image

# path = 'TrainTestData/Test/Normal'
# files = [f for f in glob.glob(path + "**/*.jpg", recursive=True)]
# f=files[0]
# img=Image.open(f)
# img=img.resize((300,300))

# print(str(f),model.predict(np.asarray(img)/255))



# # for f in files:
# #     img=Image.open(f)
# #     img=img.resize((300,300))
# #     print(str(f),model.predict(img))






