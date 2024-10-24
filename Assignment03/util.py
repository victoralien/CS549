from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
import numpy as np
import math

def Sigmoid(x):
	g = float(1.0 / float((1.0 + math.exp(-1.0*x))))
	return g

def Prediction_Intercept(theta, x):
	z = theta[0]
	for i in range(1, len(theta)):
		z += x[i-1] * theta[i]
	return Sigmoid(z)

def Cost_Function_Intercept(X,Y,theta,m):
	sumErrors = 0

	for i in range(m):
		xi = X[i]
		est_yi = Prediction_Intercept(theta,xi)
		if Y[i] == 1:
			error = -Y[i] * math.log(est_yi)
		elif Y[i] == 0:
			error = -(1 - Y[i]) * math.log(1 - est_yi)
		sumErrors += error
	return sumErrors

def Gradient_Descent_Intercept(X,Y,theta,m,alpha):
	newTheta = theta.copy()
	for j in range(len(theta)):
		sumErrors = 0
		for i in range(m):
			xi = X[i]
			if j == 0:
				xij = 1
			else:
				xij = xi[j-1]
			hi = Prediction_Intercept(theta,X[i])
			error = (hi - Y[i]) * xij
			sumErrors+=error
		deltaF = (alpha / m) * sumErrors
		newTheta[j] = theta[j] - deltaF
	return newTheta

def calConfustionMatrix(predY,trueY):
    labels = [0,1]

    cMatrix = confusion_matrix(trueY,predY,labels=labels)

    accuracy = accuracy_score(trueY,predY)
    precision = precision_score(trueY,predY,labels=labels,average=None)
    recall = recall_score(trueY,predY,labels=labels,average=None)

    return cMatrix,accuracy,precision,recall
			
def func_calConfusionMatrix(predY,trueY):
	
    TP = TN = FP = FN = 0

    for pred, true in zip(predY, trueY):
        if pred == 1 and true == 1:
            TP += 1
        elif pred == 0 and true == 0:
            TN+=1
        elif pred == 1 and true == 0:
            FP += 1
        elif pred == 0 and true == 1:
            FN+=1
    print("Confmatrix:\n[[TP,FP]\n[FN,TN]]\n[[{:},{:}]\n[{:},{:}]]".format(TP,FP,FN,TN))
	
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
	
    return  accuracy, precision, recall

def func_calConfusionMatrix_sklearn(predY,trueY):
	
    predY=predY.ravel()
    trueY=trueY.ravel()
    TP = TN = FP = FN = 0

    for pred, true in zip(predY, trueY):
        if pred == 1 and true == 1:
            TP += 1
        elif pred == 0 and true == 0:
            TN+=1
        elif pred == 1 and true == 0:
            FP += 1
        elif pred == 0 and true == 1:
            FN+=1
    print("Confmatrix:\n[[TP,FP]\n[FN,TN]]\n[[{:},{:}]\n[{:},{:}]]".format(TP,FP,FN,TN))
	
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
	
    return  accuracy, precision, recall



