import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier 
from scipy.stats import gamma 
import random

def getData(n, d):
    X = np.loadtxt('X.dat')
    Y = np.loadtxt('Y.dat')
    trainX = X[0:n,:d]
    trainY = Y[0:n]
    testX = X[1000:2000,:d]
    testY = Y[1000:2000]
    return trainX, trainY, testX, testY


def optimalROC():
    
    numSamples = 10**5

    #generate labels
    Y = 1+ (np.random.random(size=numSamples)<0.6)
    R1 = np.random.gamma(shape=20, scale=0.1, size=numSamples)
    R2 = np.sqrt(2*np.random.exponential(scale=1, size=numSamples))
    R = R1
    R[Y==2] = R2[Y==2]
    
    #Class conditional densities to get NP-classifier likelihoods
    logf1 = np.log(gamma.pdf(R,20, 0.1))
    logf2 = np.log(R)-(R**2)/2
    likelihoodRatios = logf2-logf1

    # Sort ratios and get their labels
    sortIndices = np.argsort(likelihoodRatios)
    sortedRatios = likelihoodRatios[sortIndices]
    sortedRatioLabels = Y[sortIndices]

    # Calc False Alarm Rate (FAR) and Detection Rate (DR)
    far = 1-np.cumsum(sortedRatioLabels==1)/np.sum(sortedRatioLabels==1)
    far[-1]=1
    far[-1]=0
    dr = 1-np.cumsum(sortedRatioLabels==2)/np.sum(sortedRatioLabels==2)
    dr[-1]=1
    dr[-1]=0

    plt.plot(far,dr, label='NP-Optimal', lw=2.5, color='black')


def main():

    n = 1000 # The size of training set (vary between 200 and 1000)
    dVec = [2, 5, 50, 200] # Number of features to use 
    kVec = [1, 5, 21, 101, 501] # Parameter for nearest neighbors classifier

    for d in dVec:

        # Split data into training/testing sets
        trainX, trainY, testX, testY = getData(n, d)
            
        for k in kVec:

            if k>n:
                k=n
            # Instantiate + Fit K-Nearest Neighbors Model
            knnModel = KNeighborsClassifier(n_neighbors=k) 
            knnModel.fit(trainX, trainY)

            # Get r-hat: (1000x2) matrix of predicted test set class probabilities
            rhat = knnModel.predict_proba(testX)
            r_2k = rhat[:,1] # fraction of each example's k nearestneighbors that were 2
            
            sortIndices = np.argsort(r_2k)
            sortedLabels = testY[sortIndices]

            # Calc False Alarm Rate (FAR) and Detection Rate (DR)
            far = 1-np.cumsum(sortedLabels==1)/np.sum(sortedLabels==1)
            dr = 1-np.cumsum(sortedLabels==2)/np.sum(sortedLabels==2)
            plt.plot(far,dr,label='k={}'.format(k))
        
        # Plot optimal ROC curve
        optimalROC()
        plt.title("training size={}, number of features={}".format(n,d),fontsize=16)
        plt.legend(fontsize=14)
        plt.xlabel('FAR',fontsize=14)
        plt.ylabel('DR',fontsize=14)
        plt.savefig("n={}, d={}".format(n,d))
        plt.show()

if __name__ == "__main__":
    main()