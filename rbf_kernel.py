import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.svm import SVC  
from scipy.stats import gamma 
import random

def getData(n, d):
    X = np.loadtxt('X.dat')
    Y = np.loadtxt('Y.dat')
    trainX = X[0:n,:d]
    trainY = Y[0:n]
    trainY = 2*trainY - 3
    return trainX, trainY


def make_meshgrid(x,y,h):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def main():
    #==========PARAMATERS========#
    n = 200
    d = 2
    C = 1
    C_vec = [1,5,10,10**6]
    h = 0.01
    sig_vec = [0.01,0.5, 2.0,20]
    sigma = 1.3
    # Split data into training/testing sets
    trainX, trainY = getData(n, d)

    # for sigma in sig_vec:
    for C in C_vec:
        svm = SVC(kernel="rbf",C=C, gamma=1/sigma,probability=True)
        H = svm.fit(trainX,trainY)
        X0, X1 = trainX[:,0], trainX[:,1]
        xx, yy = make_meshgrid(X0, X1, h)

        fig, ax = plt.subplots()
        
        z = svm.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
        z = z.reshape(xx.shape)
        bound = H.predict(np.c_[xx.ravel(),yy.ravel()])
        bound = bound.reshape(xx.shape)

        #===CONTOUR PLOT=====#
        ax.contourf(xx,yy,z, cmap=plt.cm.coolwarm, alpha=0.5)
        ax.contour(xx,yy,bound,colors='black', alpha=1,linewidths=2)
        ax.scatter(X0,X1, c=trainY, cmap=plt.cm.coolwarm,s=200, edgecolors='black',linewidths=1)
        fig.set_size_inches(10,10)
        plt.title('SVM boundaries: n={}, sigma={}, C={}'.format(n,sigma,C), fontsize=30)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.savefig('SVM boundaries: n={}, sigma={}, C={}.png'.format(n,sigma,C))
        plt.show()

        #====SURFACE PLOT======#
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(xx,yy,z, cmap=plt.cm.coolwarm, alpha=1)
        ax.set_zlim(z.min()-0.5, z.max()+0.5)
        fig.set_size_inches(10,10)
        plt.title('Contour plot: n={}, sigma={}, C={}'.format(n,sigma,C), fontsize=30)
        plt.savefig('Contour plot: n={}, sigma={}, C={}.png'.format(n,sigma,C))
        plt.show()



    pass

if __name__ == '__main__':
    main()