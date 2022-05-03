import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

nvec = np.array([50, 100, 150, 200, 250, 300])

# q=Pstar distribution from part a
q = np.array([0.13235832, 0.12609238, 0.23260273, 0.10385781, 0.17958991, 0.22549885])

dice = [1,2,3,4,5,6]
lookup = [1,2,3,3,4,5,5,6,6,6]
weights = [.1,.1, .2, .1, .2, .3]

mM_probRareEvent = []
logRareEvent = []
meanDistributionDistance = []
stdDistributionDistance = []


def distributionDistance(q, phat):
    dist = 0
    for x in range(6):  
        diff = np.abs(q[x]-phat[x])
        if diff > dist:  
            dist = diff
    return dist

# Run Simulations and create data
for n in nvec:

    #First, gather all 100 rare samples for each n
    
    print('\nn = ', n)
    rareSamples = []
    numRare = 0
    numSamples = 0

    while len(rareSamples) < 100:

        numSamples += 1

        sample = [lookup[np.random.randint(0,10)] for _ in range(n)]
        assert len(sample) == n
        U = np.sum([x*x for x in sample])/n

        if U < 17:
            numRare += 1
            rareSamples.append(sample)
    
    print("numSamples: ", numSamples)

    assert len(rareSamples) == 100

    #Turn Rare Samples into observed empirical distrs
    rareSamples = np.asarray(rareSamples)
    empiricalDistrs = np.asarray([np.unique(rareSamples[x], return_counts=True)[1]/n for x in range(100)])

    #turn empirical distrs into deltas 
    deltas = [distributionDistance(q, empiricalDistrs[i]) for i in range(100)]

    meanDelta = np.mean(deltas)
    stdDelta = np.std(deltas)

    #Calculate m/M
    mM = 100/numSamples
    logmM = np.log(mM)/n

    mM_probRareEvent.append(mM)
    logRareEvent.append(logmM)
    meanDistributionDistance.append(meanDelta)
    stdDistributionDistance.append(stdDelta)


# Plot data
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 13})
sns.set_style("white")
fig, axs = plt.subplots(2,2)

sns.lineplot(ax=axs[0,0], x=nvec, y=mM_probRareEvent)
axs[0,0].set_title("Probability of Rare Event")
axs[0,0].set_xlabel("n")
axs[0,0].set_ylabel(r'$\frac{m}{M}$')

sns.lineplot(ax=axs[0,1], x=nvec, y=logRareEvent)
axs[0,1].set_title("Relative Entropy")
axs[0,1].set_xlabel("n")
axs[0,1].set_ylabel(r'$\frac{1}{n}log(\frac{m}{M})$')

sns.lineplot(ax=axs[1,0], x=nvec, y=meanDistributionDistance)
axs[1,0].set_title("Mean Distances from Prediction")
axs[1,0].set_xlabel("n")
axs[1,0].set_ylabel(r'$\bar{\Delta}$')

sns.lineplot(ax=axs[1,1], x=nvec, y=stdDistributionDistance)
axs[1,1].set_title("Standard Deviation of Distances from Prediction")
axs[1,1].set_xlabel("n")
axs[1,1].set_ylabel(r'$S_{\Delta}$')

fig.tight_layout()

plt.show()


