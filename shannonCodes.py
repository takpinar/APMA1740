import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns

def main():

    p = .2
    qList = [.01, .1, .2, .5, .8]
    nList = [1, 2, 3, 4, 5, 10, 20, 100, 500]
    allLengths = []

    for q in qList:

        print("q: ", q)
        lengths = []

        for n in nList:

            print('n: ', n)
            count = 0
            runningCodeLengthSum = 0

            while count < 10**5:
                count += 1
                sample = [np.random.binomial(2, p) for _ in range(n)]
                f_nq_x = np.prod([q**sample[x] * (1-q)**(1-sample[x]) for x in range(n)])
                
                if f_nq_x == 0:
                    codeLength = 0
                else:
                    codeLength = np.ceil(-np.log2(f_nq_x))

                runningCodeLengthSum += codeLength

            avgLength = (runningCodeLengthSum/10**5) / n
            lengths.append(avgLength)   
        
        allLengths.append(lengths)
    
    
    assert len(allLengths) == 5
    plt.rcParams['text.usetex'] = True
    plt.plot(nList, allLengths[0], label = 'q = 0.01') 
    plt.plot(nList, allLengths[1], label = 'q = 0.1') 
    plt.plot(nList, allLengths[2], label = 'q = 0.2') 
    plt.plot(nList, allLengths[3], label = 'q = 0.5') 
    plt.plot(nList, allLengths[4], label = 'q = 0.8') 
    plt.legend(loc='upper right')
    plt.xlabel('n')
    plt.ylabel(r'$\frac{1}{n}E|C_{n,q}(X_{1:n})|$')
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    main()