import numpy as np
import cv2
#from features import logfbank
#import scipy.io.wavfile as wav
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.pairwise import chi2_kernel
import sys



X2=np.array([list(i)[:10000] for i in pickle.load( open( ".\\speech\\X1.pkl", "rb" ) )])

#Seperated into two files for size issues
# X1=np.array([list(i)[:326200] for i in pickle.load( open(".\\palm\\corpus\\X1.pkl", "rb" ) )])
# X2=np.append(X1,np.array([list(i)[:326200] for i in pickle.load( open( ".\\palm\\corpus\\X2.pkl", "rb" ) )]),0)
#print(len(X1))
#print(len(X2))
#X=np.array(X2)
print(len(X2))

print(sys.getsizeof(X2))

y=[]
for i in range(1,17):
	for j in range(1,11):
		y.append(i)

y=np.array(y)


skf = StratifiedKFold(y, n_folds=10)



model = SVC(kernel="poly")
model.fit(X2, y)
#del(X2)
#del(y)
#pickle.dump( model, open( "ladpalmtrain.pkl", "wb"))

scores = ['precision', 'recall']

for train_index, test_index in skf:
	for score in scores:
		print("TRAIN:", train_index, "TEST:", test_index)
 		X_train, X_test = X2[train_index], X2[test_index]
	 	y_train, y_test = y[train_index], y[test_index]

		
# 	#k = chi2_kernel(X_train, gamma=.5)
# 	#k2 =chi2_kernel(X_test, gamma=.5)

	 	model = SVC(kernel='linear')
# 	#tuned_parameters = [{'kernel': ['poly'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# 	#model= GridSearchCV(SVC(), tuned_parameters, cv=5,scoring='%s' % score)
# 	#print(model)
 		model.fit(X_train, y_train)
	 	expected = y_test
	 	predicted = model.predict(X_test)

 	# summarize the fit of the model
 		print(metrics.classification_report(expected, predicted))
 	 	print(metrics.confusion_matrix(expected, predicted))

	break
#  	#res1 = process(img, filters)
