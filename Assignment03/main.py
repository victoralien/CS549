import numpy as np
import matplotlib.pyplot as plt
from getDataset import getDataSet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from util import Gradient_Descent_Intercept, Prediction_Intercept, Cost_Function_Intercept,func_calConfusionMatrix,calConfustionMatrix

# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.

[X, y] = getDataSet()  # note that y contains only 1s and 0s,

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)


# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120

######################PLACEHOLDER 1#start#########################
# write you own code to randomly pick up nTrain number of samples for training and use the rest for testing.
# WARNIN: 

#Same function used in Quiz02
trainX, testX, trainY, testY = train_test_split(X,y,test_size=.33)

####################PLACEHOLDER 1#end#########################
# plot the samples you have pickup for training, check to confirm that both negative
# and positive samples are included.
func_DisplayData(trainX, trainY, 2, 'training samples')
func_DisplayData(testX, testY, 3, 'testing samples')
# show all charts
plt.show()

#  step 2: train logistic regression models

######################PLACEHOLDER2 #start#########################
# in this placefolder you will need to train a logistic model using the training data: trainX, and trainY.

theta_intercept = [0,0,0]
alpha = 0.1
max_iteration = 1000

for x in range(max_iteration):
    theta_intercept = Gradient_Descent_Intercept(trainX,trainY,theta_intercept,len(trainX),alpha)

######################PLACEHOLDER2 #end #########################
# step 3: Use the model to get class labels of testing samples.
######################PLACEHOLDER3 #start#########################
# codes for making prediction, 
# with the learned model, apply the logistic model over testing samples
# hatProb is the probability of belonging to the class 1.
# y = 1/(1+exp(-Xb))
# yHat = 1./(1+exp( -[ones( size(X,1),1 ), X] * bHat )); ));
# yHat = Prediction_Intercept(theta_intercept,testX)
# yHat = (yHat >= 0.5).astype(int)

hatProb = []

for i in range(len(testX)):
    prediction = round(Prediction_Intercept(theta_intercept,testX[i]))
    hatProb.append(prediction)

######################PLACEHOLDER 3 #end #########################
# step 4: evaluation
# compare predictions yHat and and true labels testy to calculate average error and standard deviation

yHat = []

for i in range(len(hatProb)):
    if(hatProb[i] >= .5):
        yHat.append(1)
    else:
        yHat.append(0)

accuracy, precision, recall = func_calConfusionMatrix(yHat, testY)

testYDiff = np.abs(hatProb - testY)

avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)

print('Average error: {:.4f} (Standard Deviation: {:.4f})'.format(avgErr,stdErr))



# Print the confusion matrix and performance metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
