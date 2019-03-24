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

X = []
y = []

#	Feature extraction using mfcc features

a="C:\\Project\\speech\\corpus\\"
for num in range(1,6):
	for ite in range(1,11):
		(rate,sig) = wav.read(a+str(num)+"\\"+str(ite)+".wav")
		mfcc_feat = mfcc(sig,rate)
		print(len(logfbank(sig,rate).flatten()))
		X.append(logfbank(sig,rate)[:10000,:].flatten())
		print(num,ite)

		#y.append(num)
pickle.dump( X, open( "XX.pkl", "wb" ))
		#pickle.dump( y, open( "y.pkl", "wb" ))
		# print(fbank_feat[:500,:].flatten())
