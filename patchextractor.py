import numpy as np 
import cv2 as cv
import os                                                                    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
X=[]
Y=[]
Z=[1,4,5,6,7]
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
	X=X+patchextractor(img,(128,128),64)
	#print(len(X))
	img=cv.imread(inY,0)
	Y=Y+patchextractor(img,(128,128),64)
	#print(len(Y))
	#print("itertation%d complete"%(k))
	
	

X=np.array(X)
Y=np.array(Y)
X=np.expand_dims(X,-1)
Y=np.expand_dims(Y,-1)



