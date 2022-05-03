import numpy as np

def entropy(p):
    ent = -np.sum([px*np.log2(px) for px in p])
    return ent

p = [.1, .1, .1, .7]
print(entropy(p))