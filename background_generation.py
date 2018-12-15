import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

Z=np.random.multivariate_normal(gmm.params.mu[0],gmm.params.Sigma[0],(X.shape[0],X.shape[1])).astype("float32")
Z=cv.cvtColor(Z,cv.COLOR_BGR2RGB)
plt.imsave('background0_1',Z)
Z=np.random.multivariate_normal(gmm.params.mu[1],gmm.params.Sigma[1],(X.shape[0],X.shape[1])).astype("float32")
Z=cv.cvtColor(Z,cv.COLOR_BGR2RGB)
plt.imsave('background1_1',Z)
Z=np.random.multivariate_normal(gmm.params.mu[2],gmm.params.Sigma[2],(X.shape[0],X.shape[1])).astype("float32")
Z=cv.cvtColor(Z,cv.COLOR_BGR2RGB)
plt.imsave('background2_1',Z)
Z=np.random.multivariate_normal(gmm.params.mu[3],gmm.params.Sigma[3],(X.shape[0],X.shape[1])).astype("float32")
Z=cv.cvtColor(Z,cv.COLOR_BGR2RGB)
plt.imsave('background3_1',Z)
