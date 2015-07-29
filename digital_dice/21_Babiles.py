import numpy as np

def prob(k):
    if k==0:
        return 0.4825
    else:
        return 0.2126*0.5893**(k-1)
