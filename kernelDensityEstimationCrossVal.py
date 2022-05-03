import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib.patheffects as pe
from scipy.stats import norm


def phi_plot(x, l):

    mean = l
    std = np.abs(np.cos(l))
    phi = norm(loc=mean, scale=std).pdf(x)
    # plt.plot(x,phi)
    # plt.show()

    return norm(loc=mean, scale=std).pdf(x)


def f_plot(hold):

    x = np.linspace(-2, 14, num=16000)
    f = np.divide(np.sum([phi_plot(x, l) for l in np.arange(1, 11)], axis=0) ,10)
    plt.plot(x,f, lw = 2,color = 'k', path_effects=[pe.Stroke(linewidth=5, foreground='g'), pe.Normal()])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if hold:
        return f
    else:
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.show()


def f_sample(n):

    #uniform sample (1,10)
    L_n = np.random.randint(1,10, size=n)
    samples = np.array([np.random.normal(loc=l, scale = np.abs(np.cos(l))) for l in L_n])
    # plt.hist(samples, bins = 1000)
    # plt.show()
    return samples


def kernel(mu, w, x):
    return norm(mu, w).pdf(x)

def kernelRoot2(mu, w, x):
    return norm(mu, np.sqrt(2)*w).pdf(x)


def density_estimator(n, h, samples, hold):
    #Estimate density f using n data samples and fixed bandwidth h

    x = np.linspace(-2, 14, num=16000)

    fhat = np.divide(np.sum([kernel(mu, h, x) for mu in samples], axis=0),n)
    f=f_plot(hold=True)

    ise_reimann = np.sum([np.square(np.abs(fhat[xi]-f[xi]))*0.001 for xi in range(len(f))])
    

    plt.plot(x,fhat, lw=1.2, label = 'Omega = {}'.format(round(h, 2)))
    
    if not hold:
        fig = plt.gcf()
        fig.set_size_inches(20, 10.5)        
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Cross Validated f-Hat Kernel Density Estimator', fontsize=25)
        plt.legend(prop={'size': 23})
        plt.savefig('crossValidatedFhat.png')
        plt.show()

    return ise_reimann, fhat
        

def density_estimators50(n):
    
    x = np.linspace(-2, 14, num=16000)
    mise_est = 0
    min_h = np.inf
    max_h = -np.inf
    avg_h = 0
    for est in range(50):

        print(est)
        #Plot 
        h, samples = crossValidatedKernelDensityEstimator(n, hold=True)
        ise_reimann, fhat = density_estimator(n, h, samples, hold=True)
        plt.plot(x,fhat, lw=1.7) 

        mise_est += ise_reimann
        if h<min_h:
            min_h = h
        if h>max_h:
            max_h = h
        avg_h += h

    f=f_plot(hold=True)
    mise_est= mise_est/50
    avg_h = avg_h/50
    print('\nMISE  = {}'.format(mise_est))
    print("Max Bandwidth = {}".format(max_h))
    print("Min Bandwidth = {}".format(min_h))
    print("Average Bandwidth = {}".format(avg_h))
    fig = plt.gcf()
    fig.set_size_inches(20, 10.5)
    plt.title('50 Cross Validated Kernel Density Estimators', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('50crossVal.png')
    plt.show()


def crossValidatedKernelDensityEstimator(n, hold):

    
    samples = f_sample(n)
    D = np.abs(np.subtract.outer(samples, samples))

    #Optimize J
    min_jHat = np.inf
    argmin_h = None
    hvec = np.arange(0.01,2,0.01)
    jHatVec = np.zeros([199])
    for i, h in enumerate(hvec):

        term1 = 2*kernel(0, h, 0)/(n-1)

        kRoot = kernelRoot2(0, h, D)
        k2 = ((2*n)/(n-1))*kernel(0, h, D)
        term2 = (1/n**2)*np.sum(np.sum(kRoot-k2))

        jHat = term1 + term2
        jHatVec[i] = jHat
        if jHat < min_jHat:
            min_jHat = jHat
            argmin_h = h

    print('argmin h: ',argmin_h)
    print('jHat: ', jHat, '\n')

    if hold==False:
        plt.plot(hvec, jHatVec, lw=1.4)
        plt.xlabel('Bandwidth', fontsize=22)
        plt.ylabel('j-Hat', fontsize = 22)
        plt.title('j-Hat vs Bandwidth', fontsize=25)
        fig = plt.gcf()
        fig.set_size_inches(20, 10.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig('jhatVSbandwidth.png')
        plt.show()

    return argmin_h, samples


def main():

   
    print("PART B")
    h, samples = crossValidatedKernelDensityEstimator(n=200, hold=False)
    ise, _ = density_estimator(200, h, samples, hold=False)
    print("integrated square error: ", ise)

    print('\nPART C')
    density_estimators50(200)
    # c) d)
    # print("PART C")
    # density_estimators50()

if __name__ == "__main__":
    main()