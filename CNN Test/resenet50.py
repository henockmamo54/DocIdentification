# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:45:07 2020

@author: Henock
"""


# example of loading the resnet50 model
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator


# load model
model = ResNet50(weights="imagenet")
# summarize the model
model.summary()


# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(512, activation='relu')(flat1)
output = Dense(1, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()


# mark loaded layers as not trainable
for layer in model.layers[:-3]:
	layer.trainable = False
    
# for i, layer in enumerate(model.layers):
#     print(i,layer.name,"-",layer.trainable)
    

batch_size = 30


train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Testing Augmentation - Only Rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)

# Generates batches of Augmented Image data
train_generator = train_datagen.flow_from_directory('../TrainTestData/Train', target_size = (300, 300), 
                                                    batch_size = batch_size,
                                                    class_mode = 'binary') 

# Generator for validation data
validation_generator = test_datagen.flow_from_directory('../TrainTestData/Test', 
                                                        target_size = (300, 300),
                                                        batch_size = batch_size,
                                                        class_mode = 'binary')



model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])


# Fit the model on Training data
history = model.fit_generator(train_generator,
                    epochs = 10,
                    validation_data = validation_generator,
                    verbose = 1)



# Evaluating model performance on Testing data
loss, accuracy = model.evaluate(validation_generator)

print("\nModel's Evaluation Metrics: ")
print("---------------------------")
print("Accuracy: {} \nLoss: {}".format(accuracy, loss))



from matplotlib import pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("resnet50_accuracy.png")
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("resnet50_loss.png")
plt.show()




    

