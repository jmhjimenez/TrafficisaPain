from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
from skimage.io import imread
import random

bs = 8

img_y = open('../data/manual_03.ntxy', 'r')
img_y = list(img_y)[:128]

x = []
y = []

for line in img_y:
    try:
        _, img, yx, yy = line.split(' ')
        img = imread('../data/resized/03'+img+'.jpg')
        x.append(img)
        y.append([int(yx)//5, int(yy)//5])
    except:
        pass

images = np.array(x)
images = np.divide(images, 255.0)
positions = np.array(y)



base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(162, 216, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# x, y
x = Dense(2, activation='linear')(x)
model = Model(inputs= base_model.input, outputs=x)


# def loss_function(y_true, y_pred):
#     # mse on x, y
#     # categorical cross entropy on p

model.compile(loss='mse', optimizer='adam')

img_train = images[:100]
img_test = images[100:]
pos_train = positions[:100]
pos_test = positions[100:]

model.fit(img_train, pos_train, validation_data=(img_test, pos_test), batch_size=bs, epochs=20)

y_pred = model.predict(img_train[:10])
print(pos_train[:10])
print(y_pred)
