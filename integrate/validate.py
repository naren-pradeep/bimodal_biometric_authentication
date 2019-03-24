import numpy as np
import cv2
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
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
 


#X1=np.array([list(i)[:666000] for i in pickle.load( open(".\\palm\\corpus\\X1.pkl", "rb" ) )])
#X2=np.append(X1,np.array([list(i)[:666000] for i in pickle.load( open( ".\\palm\\corpus\\X2.pkl", "rb" ) )]),0)
# #print(len(X1))
# #print(len(X2))
# X=np.array(X2)


print ("processing  palm samples.....")
y=[]
for i in range(1,17):
	for j in range(1,11):
		y.append(i)

y=np.array(y)


img = color.rgb2gray(cv2.imread(".\\palm\\corpus\\temp.jpg"))
filters = build_filters()
 
res1 = process(img, filters)
res1=np.array(res1.flatten())

# model = SVC(kernel="linear")

# model.fit(X, y)
model=pickle.load( open( "palmtrain.pkl", "rb" ))
predicted = model.predict(res1)

print("The predicted result from palm :")
print(predicted)
		

print("processing speech samples")
X1=np.array([list(i)[:10000] for i in pickle.load( open( ".\\speech\\XX.pkl", "rb" ) )])

y=[]
 #y = np.array(pickle.load( open( "yy.pkl", "rb" ) ))
for i in range(1,17):
	for j in range(1,10):
 		if i != predicted:
 			y.append(0)
  		else:
 			y.append(1)	

y=np.array(y)

X1_test=[]
(rate,sig) = wav.read("s.wav")
mfcc_feat = mfcc(sig,rate)
#print(len(X1))
#print(len(logfbank(sig,rate)[:10000].flatten()))
X1_test=(logfbank(sig,rate).flatten()[:10000])
X1_test=np.array(X1_test)
model = SVC(kernel="linear")
model.fit(X1, y)
ans = model.predict(X1_test)

print("the prediction from speech :")
if(ans==1):
	print(predicted)
else:
	print("not")

if(ans==1):	
 	print("Validated")
else:
 	print("not validated")