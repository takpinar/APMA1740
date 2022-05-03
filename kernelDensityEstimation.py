import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib.patheffects as pe
from scipy.stats import norm

data = np.random.normal()


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

def kernelW(mu, w, x):
    return norm(mu, w**2).pdf(x)

def density_estimators(samples):

    n = samples.size
    omegas = np.array([0.25, 0.5, 1.1])
    x = np.linspace(-2, 14, num=16000)

    for i, w in enumerate(omegas):
        
        fhat = np.divide(np.sum([kernelW(mu, w, x) for mu in samples], axis=0),n)
        f=f_plot(hold=True)
        plt.plot(x,fhat, lw=2, label = 'Omega = {}'.format(w))
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.legend(prop={'size': 23})
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.savefig('Omega = {}.png'.format(w))
        plt.show()
        

def density_estimators50():
    
    
    omegas = np.array([0.25, 0.5, 1.1])
    x = np.linspace(-2, 14, num=16000)

    for i, w in enumerate(omegas):
        
        mise_est = 0

        for _ in range(50):

            samples = f_sample(200)
            n = samples.size
            #Plot 
            fhat = np.divide(np.sum([kernelW(mu, w, x) for mu in samples], axis=0),n)
            plt.plot(x,fhat, lw=1) 
            f=f_plot(hold=True)

            #Compute ISE using Reimann Sum
            ise_reimann = np.sum([np.square(np.abs(fhat[xi]-f[xi]))*0.001 for xi in range(len(f))])
            mise_est += ise_reimann
            
        mise_est= mise_est/50
        print('\nMISE for Omega = {}: {}'.format(w,mise_est))
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.title('Omega = {}'.format(w), fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig('Omega50 = {}.png'.format(w))
        plt.show()
        
def main():

    # a) Plot f(x)
    f_plot(hold=False)

    # b) 
    samples = f_sample(200)
    density_estimators(samples)

    # c) d)
    density_estimators50()

if __name__ == "__main__":
    main()