import numpy as np
from matplotlib import pyplot as plt
import math


def sampleNormal(s,n,r):
    Y = np.random.normal(0,1,size=(s,n))
    normalizer = np.sqrt(np.sum(np.square(Y), axis=-1))
    
    X = r * (Y/normalizer[:,None])
    return X

def sampleAngles(n,r):
    sample = np.random.uniform(0, 2*math.pi, n)
    return r*np.cos(sample), r*np.sin(sample)

def main():
    firstSample = sampleNormal(200, 2, 5)
    plt.scatter(firstSample[:,0], firstSample[:,1])
    

    angleSampleX, angleSampleY = sampleAngles(200, 5)
    plt.scatter(angleSampleX, angleSampleY)
    plt.show()
    

if __name__ == "__main__":
    main()