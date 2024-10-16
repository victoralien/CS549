"""
This scripts includes two types of implementations of logistric regression. The first one is to implement the gradient descent (GD) method from scratch; the other is to call the sklearn library to do the same thing. 

The scripts are from the open source community.

It will also compare how these two methods work to predict the given outcome
for each input tuple in the datasets.
 
"""

import math
import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

# import self-defined functions
from util import Cost_Function, Gradient_Descent, Cost_Function_Derivative, Cost_Function, Prediction, Sigmoid, Gradient_Descent_Intercept,miniGradient_Descent_Intercept,Prediction_Intercept,Cost_Function_Intercept

########################################################################
########################### Step-1: data preprocessing #################
########################################################################

# scale data to be between -1,1 

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
df = pd.read_csv("data.csv", header=0)

# clean up data
df.columns = ["grade1","grade2","label"]

x = df["label"].map(lambda x: float(x.rstrip(';')))

# formats the input data into two arrays, one of independant variables
# and one of the dependant variable
X = df[["grade1","grade2"]]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"].map(lambda x: float(x.rstrip(';')))
Y = np.array(Y)

print(X.shape)
print(Y.shape)


# save the data in
##X = pd.DataFrame.from_records(X,columns=['grade1','grade2'])
##X.insert(2,'label',Y)
##X.to_csv('data2.csv')

########################################################################
########################### Step-2: data splitting #################
########################################################################
# split the dataset into two subsets: testing and training
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)

########################################################################
#################Step-3: training and testing using sklearn    #########
########################################################################

# use sklearn class
clf = LogisticRegression()
# call the function fit() to train the class instance
clf.fit(X_train,Y_train)
# scores over testing samples
print(clf.score(X_test,Y_test))

'''
# visualize data using functions in the library pylab 
pos = where(Y == 1)
neg = where(Y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature 1: score 1')
ylabel('Feature 2: score 2')
legend(['Label:  Admitted', 'Label: Not Admitted'])
show()
'''


########################################################################
##############Step-4: training and testing using self-developed model ##
########################################################################

#

theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations
m = len(Y) # number of samples

for x in range(max_iteration):
    # call the functions for gradient descent method
    new_theta = Gradient_Descent(X,Y,theta,m,alpha)
    #theta = miniDescent(X,Y,theta,m,alpha,batchSize)
    theta = new_theta
    if x % 200 == 0:
        # calculate the cost function with the present theta
        Cost_Function(X,Y,theta,m)
        print('OriginalTheta ', theta)
        print('OrigninalCost is ', Cost_Function(X,Y,theta,m))

theta_intercept = [0,0,0]
for x in range(max_iteration):
    theta_intercept = Gradient_Descent_Intercept(X_train,Y_train, theta_intercept, len(Y_train),alpha)
    if x % 200 == 0:
        print(f'GDIntercept {x}: Cost {Cost_Function_Intercept(X_train, Y_train, theta_intercept, len(Y_train))}, theta: {theta_intercept}')


#Mini batch testing
theta_intercept_mb = [0,0,0]
batchSize = 20

for x in range(max_iteration):
    theta_intercept_mb = miniGradient_Descent_Intercept(X,Y,theta_intercept_mb,m,alpha,batchSize)
    if x % 200 == 0:
        cost = Cost_Function_Intercept(X,Y,theta_intercept_mb,m)
        print('MiniBatchTheta ', theta_intercept_mb)
        print('MiniBatchCost is ', Cost_Function_Intercept(X,Y,theta_intercept_mb,m))


########################################################################
#################         Step-5: comparing two models         #########
########################################################################
##comparing accuracies of two models. 
score = 0
for i in range(len(X_test)):
    prediction = round(Prediction(X_test[i],theta))
    answer = Y_test[i]
    if prediction == answer:
        score += 1
accuracy_og = float(score) /len(X_test)

score_mb = 0
for i in range(len(X_test)):
    prediction = round(Prediction_Intercept(theta_intercept_mb, X_test[i]))
    if prediction == Y_test[i]:
        score_mb += 1
accuracy_mb = float(score_mb) / len(X_test)

score_intercept = 0
for i in range(len(X_test)):
    prediction = round(Prediction_Intercept(theta_intercept, X_test[i]))
    if prediction == Y_test[i]:
        score_intercept += 1
accuracy_intercept = float(score_intercept) / len(X_test)

scikit_score = clf.score(X_test,Y_test)
print("\nAccuracy Comparison:")
print(f'Scikit-learn Logistic Regression accuracy: {scikit_score * 100:.2f}%')
print(f'Mini-batch Gradient Descent (batch size {batchSize}) accuracy: {accuracy_mb * 100:.2f}%')
print(f'Gradient Descent with Intercept accuracy: {accuracy_intercept * 100:.2f}%')
print(f'Gradient Descent w/o Intercept Accuracy: {accuracy_og * 100:.2f}%')

'''
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(X_test,Y_test)
length = len(X_test)
for i in range(length):
    prediction = round(Prediction(X_test[i],theta))
    answer = Y_test[i]
    if prediction == answer:
        score += 1
my_score = float(score) / float(length)
if my_score > scikit_score:
    print('You won!')
elif my_score == scikit_score:
    print('Its a tie!')
else:
    print('Scikit won.. :(')
print('Your score: ', my_score)
print('Scikits score: ', scikit_score)
'''
