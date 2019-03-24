#from features import mfcc
#from features import logfbank
import scipy.io.wavfile as wav
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV



X=np.array([list(i)[:10000] for i in pickle.load( open( ".\\speech\\XX.pkl", "rb" ) )])


y=[]
#y = np.array(pickle.load( open( "yy.pkl", "rb" ) ))
for i in range(1,17):
	for j in range(1,10):
		#if i != var:
			y.append(i)

X=np.array(X)
y=np.array(y)

print(len(X))
print(X)

# model = SVC(kernel="poly")
# model.fit(X, y)
# pickle.dump( model, open( "palmtrain.pkl", "wb" ))

skf = StratifiedKFold(y, n_folds=5)

#scores = ['precision', 'recall']

for train_index, test_index in skf:
#for score in scores:
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	#print(X_train)



	model = SVC(kernel="poly")
	#tuned_parameters = [{'kernel': ['poly'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

	#model= GridSearchCV(SVC(), tuned_parameters, cv=5,scoring='%s' % score)
	#print(model)
	model.fit(X_train, y_train)

	# make predictions
	expected = y_test
	predicted = model.predict(X_test)

	# summarize the fit of the model
	print(metrics.classification_report(expected, predicted))
	print(metrics.confusion_matrix(expected, predicted))

	break
