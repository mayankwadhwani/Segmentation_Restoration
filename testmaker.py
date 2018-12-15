import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

def patchextractor(img,patchsize,stride):     
	patch=[]
	for i in range(0,img.shape[0]-patchsize[0]-1,stride):
	    for j in range(0,img.shape[1]-patchsize[1]-1,stride):
	        patch.append(img[i:i+patchsize[0],j:j+patchsize[1]])
	return patch

def reconstruct(patches,patch_size,imgsize,stride):     
	a,b=imgsize
	R=np.zeros((a,b)).astype("float32")
	C=np.zeros((a,b)).astype("float32")
	k=len(patches)-1
	c=0
	p=(a-patch_size[0])
	q=(b-patch_size[1])
	for i in range(0,p-1,stride):
		for j in range(0,q-1,stride):
			R[i:i+patch_size[0],j:j+patch_size[1]]+=patches[c]
			C[i:i+patch_size[0],j:j+patch_size[1]]+=1.0
			c=c+1
	R[C>0]=R[C>0]/C[C>0]
	R=R.reshape(imgsize)
	return R
"""
X1=[]
im1 = cv.imread('dibco_jpg/test/1.jpg',0)/255.0
'''im1=cv.bitwise_not(im1)
hsv = cv.cvtColor(im1, cv.COLOR_BGR2HSV) 
for x in range(0, len(hsv)):
    for y in range(0, len(hsv[0])):
        hsv[x, y][2] = 255-hsv[x, y][2]
im1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
im1=im1/255.0'''
p=im1.shape
X1=patchextractor(im1,(256,256),10)
X2=np.array(X1)
X2=np.expand_dims(X1,-1)
Y1=model.predict(X2)
print(X2.shape)
r=reconstruct(Y1[:,:,:,0],(256,256),p,10)
plt.imshow(r,cmap='gray')
plt.imsave("dibco_jpg/result/image1",r,cmap='gray')
plt.show()
"""
