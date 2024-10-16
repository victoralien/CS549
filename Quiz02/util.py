import math
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

##implementation of sigmoid function
def Sigmoid(x):
	g = float(1.0 / float((1.0 + math.exp(-1.0*x))))
	return g


##Prediction function
def Prediction(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)

def Prediction_Intercept(theta, x):
	z = theta[0]
	for i in range(1, len(theta)):
		z += x[i-1] * theta[i]
	return Sigmoid(z)

def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		est_yi = Prediction(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(est_yi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-est_yi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	#print 'cost is ', J 
	return J

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

# gradient components called by Gradient_Descent()

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Prediction(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

# execute gradient updates over thetas
def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)):
		deltaF = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - deltaF
		new_theta.append(new_theta_value)
	return new_theta


	newTheta = theta.copy()
	batchIndices = random.sample(range(m), batchSize)

	for j in range(len(theta)):
		sumErrors = 0
		for i in batchIndices:
			xi = X[i]
			if j == 0:
				xij = 1
			else:
				xij = xi[j-1]
			hi = modPrediction(theta,X[i])
			error = (hi - Y[i] * xij)
			sumErrors += error

		constant = float(alpha) / float(batchSize)
		deltaF = constant * sumErrors

		newTheta[j] = theta[j] - deltaF
	return newTheta

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

def miniGradient_Descent_Intercept(X,Y,theta,m,alpha,batchSize):
	newTheta = theta.copy()
	batchIndices = random.sample(range(m),batchSize)

	for j in range(len(theta)):
		sumErrors = 0
		for i in batchIndices:
			xi = X[i]
			if j == 0:
				xij = 1
			else:
				xij = xi[j-1]
			hi = Prediction_Intercept(theta,X[i])
			error = (hi - Y[i]) * xij
			sumErrors+=error
		deltaF = (alpha / batchSize) * sumErrors
		newTheta[j] = theta[j] - deltaF

	return newTheta
