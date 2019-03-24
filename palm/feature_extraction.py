# -*- coding: utf-8 -*-
import numpy as np
import cv2
#from features import logfbank
#import scipy.io.wavfile as wav
import PIL
from PIL import Image
from skimage.transform import resize
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from skimage import color

def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
 	kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
 	kern /= 1.5*kern.sum()
 	filters.append(kern)
 	#print len(filters)
 return filters
 
def process(img, filters):
 accum = np.zeros_like(img)
 for kern in filters:
 	fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
 	np.maximum(accum, fimg, accum)
 	#	print len(accum.flatten())
 	#X.append(accum[:800,:].flatten())#X=X.append(X,np.arraylist[:800]accum])
 return accum
 
 
if __name__ == '__main__':
	import sys
 
#print __doc__
X=[]

a="C:\\Project\\integrate\\palm\\corpus\\des\\"

for i in range (10,16):
	for j in range (1,11):
		img_fn = a+str(i)+"\\"+str(j)+".jpg"
 
		#img = cv2.imread(img_fn)
		img = color.rgb2gray(cv2.imread(img_fn))
		filters = build_filters()
 
		res1 = process(img, filters)
		res1=np.array(res1.flatten(),dtype=np.float32)
		print(len(res1))
		#res1=np.array(res1[:2750760])
		print(len(res1))
		#print((res1[:2750760].flatten()))
		X.append(res1)

X=np.array(X)


pickle.dump( X, open( "X1.pkl", "wb" ))