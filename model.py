import numpy as np
import matplotlib.pyplot as plt 
import cv2
import csv
import keras
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import tensorflow as tf

#print(keras.__version__)

##### READ DATA FILE  AND STORE IMAGES AND LABELS
def generator():
    
    lines=[]
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    del (lines[0])
    images = []
    measurements=[]
    correction = [0, 0.3, -0.3]
    for line in lines:
        #if(abs(float(line[3]))) < 0.01:
        #   continue
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/' + filename

            img = cv2.imread(current_path)
            images.append(img)
            measurement = float(line[3]) + correction[i]
            measurements.append(measurement)

            aug_img = cv2.flip(img,1)
            aug_measurement = measurement * -1
            images.append(aug_img)
            measurements.append(aug_measurement)
       
    lines=[]
    with open('new_data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    del (lines[0])
    for line in lines:
        break
       # if(abs(float(line[3]))) < 0.01:
       #     continue
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = 'new_data/IMG/' + filename

            img = cv2.imread(current_path)
            images.append(img)
            measurement = float(line[3]) + correction[i]
            measurements.append(measurement)

            #aug_img = cv2.flip(img,1)
            #aug_measurement = measurement * -1
           #images.append(aug_img)
           #measurements.append(aug_measurement) 
    return (images,measurements)


### MODEL ARCHITECTURE
def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    #model.add(Lambda(lambda x: (0.33 * x[:,:,:,:1]) + (0.33 * x[:,:,:,1:2]) + (0.33 * x[:,:,:,-1:])))
    
    model.add(Conv2D(8, (3, 3), activation="relu"))
    model.add(Conv2D(16,(3, 3), activation="relu"))
    model.add(Conv2D(24,(3, 3), activation="relu"))    
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32,(3, 3), activation="relu"))
    model.add(Conv2D(48,(3, 3), activation="relu"))
    model.add(Conv2D(56,(3, 3), activation="relu"))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(96, (3, 3), activation="relu"))
    model.add(Conv2D(128,(3, 3), activation="relu"))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = "tanh"))
              
    return model
     
#model  = keras.applications.xception.Xception(include_top=True, weights=None, input_tensor=None, input_shape=(100,100,3), pooling=None, classes=1)

#model = keras.applications.mobilenet.MobileNet(input_shape=(160,320,3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights= None, input_tensor=None, pooling=None, classes=1)

(images, measurements) = generator()
                                
#images.extend(new_images)
#measurements.extend(new_measurements) 

X_train = np.array(images)
y_train = np.array(measurements)

          
model = build_model()
model.compile(loss = "mse", optimizer = "adam")  #Compile
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs=1) #Train
model.save("modelv3.h5") #Save model
print("Model is saved")