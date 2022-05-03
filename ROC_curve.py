    import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import random
import math


##############################################
#a) generate 1000 samples
##############################################

def uniformAngle():
    return np.random.uniform(0, 2*math.pi)


def gamma():
    return np.random.gamma(20, 0.1)

def exponential():
    return np.random.exponential(1)


def sample1000():
    class1 = []
    class2 = []

    for _ in range(1000):
        rand=random.random()
        U = uniformAngle()
        if rand<0.4:
            R = gamma()
            class1.append([R*np.cos(U), R*np.sin(U)])
        else:
            R = np.sqrt(2*exponential())
            class2.append([R*np.cos(U), R*np.sin(U)])
        
    return np.asarray(class1), np.asarray(class2)

def plotSamples(class1, class2):
    fig1, ax1 = plt.subplots(2,2)

    ax1[0,0].scatter(class1[:,0], class1[:,1])
    ax1[0,0].figure.set_size_inches(6,6)
    ax1[0,0].title.set_text('X|Y=1')
    ax1[0,1].scatter(class2[:,0], class2[:,1], color='orange')
    ax1[0,1].figure.set_size_inches(6,6)
    ax1[0,1].title.set_text('X|Y=2')
    ax1[1,0].scatter(class1[:,0], class1[:,1], alpha=0.6)
    ax1[1,0].scatter(class2[:,0], class2[:,1], color='orange', alpha=0.6)
    ax1[1,0].figure.set_size_inches(6,6)
    ax1[1,0].title.set_text('X for any Y')
    fig1.delaxes(ax1[1,1])
    fig1.set_size_inches(10, 10)
    fig1.savefig('scatters.png')
    plt.show()

def main():
    class1, class2 = sample1000()
    plotSamples(class1, class2)
    

if __name__ == "__main__":
    main()

