import cv2 as cv
import numpy as np

im1=cv.imread("/home/ecsuiplab/mayank/src/6.jpg")
im1gray=cv.imread("/home/ecsuiplab/mayank/src/6.jpg",0)
im2=cv.imread("/home/ecsuiplab/mayank/src/image",0)
im3=cv.imread("/home/ecsuiplab/mayank/src/background2.jpg")
'''
a=[]
for i in range(im1[:,:,0].shape[0]-1):
    for j in range (im1[:,:,0].shape[1]-1):
        if im2[i,j]<=245:
            a.append(im1[i,j])

a=np.array(a)
p=np.mean(a,axis=0)
'''
Z=cv.resize(im3,(im1gray.shape[1],im1gray.shape[0]))
for i in range(im1gray.shape[0]-1):
    for j in range (im1gray.shape[1]-1):
        if im2[i,j]<=245:
            Z[i,j]=im1[i,j]
        elif im1gray[i,j] >=245:
            Z[i,j]=[255,255,255]
cv.imwrite("finalmaybefor6.jpg",Z)

