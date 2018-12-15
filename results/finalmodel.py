import numpy as np 
import cv2 as cv
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from keras.models import Model, load_model
from keras.layers import Input, Dense
import math
from keras.layers import Conv2D, Conv2DTranspose,Dense, Dropout, Flatten, Input, MaxPooling2D, UpSampling2D
from keras.applications.vgg16 import VGG16
from keras_contrib.losses import DSSIMObjective                                                              
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

X=[]
Y=[]
Z=[1,2,4,5,6,7,8,9,10,11]
k=0
def patchextractor(img,patchsize,stride):     
	patch=[]
	for i in range(0,img.shape[0]-patchsize[0]-1,stride):
	    for j in range(0,img.shape[1]-patchsize[1]-1,stride):
	        patch.append(img[i:i+patchsize[1],j:j+patchsize[0]])
	return patch
for i in Z:
	k+=1
	inX="../data/X/image%d.jpg"%(i)
	inY="../data/Y/image%d.jpg"%(i)	
	img=cv.imread(inX,0)
	X=X+patchextractor(img,(256,256),50)
	#print(len(X))
	img=cv.imread(inY,0)
	Y=Y+patchextractor(img,(256,256),50)
	#print(len(Y))
	#print("itertation%d complete"%(k))
	
	

X=np.array(X)
Y=np.array(Y)
X=np.expand_dims(X,-1)
Y=np.expand_dims(Y,-1)
X=X.astype("float32")/255.0
Y=Y.astype("float32")/255.0

# autoencoder using CNN for face reconstruction generate image from a given image

I=Input(shape=(256,256,1))
actFunc="relu"

#model 5s
x=Conv2D(32,(8,8),name='conv1',activation=actFunc, padding='valid',strides=(2,2))(I)
# x1= MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(x)actFunc

x=Conv2D(64,(5,5),name='conv2',activation=actFunc, padding='valid',strides=(2,2))(x)#32
# x= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='valid')(x)

x=Conv2D(128,(3,3),name='conv3',activation=actFunc, padding='valid',strides=(2,2))(x)#16
# x= MaxPooling2D(pool_size=(2, 2), strides=(2,2),padding='valid')(x)

x=Conv2D(256,(2,2),name='conv4',activation=actFunc, padding='valid',strides=(2,2))(x)#8
# x= MaxPooling2D(pool_size=(2, 2), strides=(2sigmoid,2),padding='valid')(x)

#x=Conv2D(256,(5,5),name='conv5',activation=actFunc, padding='valid',strides=(2,2))(x)#1
# x= MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(x)
y4=x

x= Conv2DTranspose(128,(4,4),activation=actFunc, padding='valid',strides=(2,2))(x)
# x=Conv2D(256,(3,3),activation='tanh', padding='valid')(x)
x= Conv2DTranspose(64,(2,2),activation=actFunc, padding='valid',strides=(2,2))(x)
# x=Conv2D(128,(3,3),activation='tanh', padding='valid')(x)
x= Conv2DTranspose(64,(2,2),activation=actFunc, padding='valid',strides=(2,2))(x)
# x=Conv2D(64,(3,3),activation='tanh', paddactFuncg='valid')(x)
x= Conv2DTranspose(16,(1,1),activation=actFunc, padding='same',strides=(2,2))(x)
x= Conv2DTranspose(8,(2,2),activation=actFunc, padding='same',strides=(1,1))(x)
# x=Conv2D(64,(3,3),activation='tanh', paddactFuncg='valid')(x)
x= Conv2DTranspose(1,(1,1),activation="sigmoid", padding='same',strides=(1,1))(x)
# x=Conv2D(1,(3,3),activation='sigmoid', padding='valid')(x)
#x= Conv2DTranspose(1,(2,2),activation='sigmoid', padding='valid',strides=(1,1))(x)

model = Model(inputs = [I], outputs= [x])
model.summary()
#model=load_model("/home/ecsuiplab/mayank/src/goodmodel_1.h5",custom_objects={ 'DSSIMObjective': DSSIMObjective(kernel_size=23)})
#model.compile(optimizer = 'rmsprop', loss=DSSIMObjective(kernel_size=23))

#history4= model.fit(X,Y,batch_size=5,epochs=10)
