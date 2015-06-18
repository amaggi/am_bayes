import numpy as np

# number of starting coins for each player
coins_init = np.array([3, 3, 3])

# probability of heads
p = 0.4

ngames = 1000
rounds = np.empty(ngames, dtype=float)

def play():

    coins = np.empty(len(coins_init), dtype=int)
    coins[:] = coins_init[:]
    nrounds = 0
    while np.min(coins) > 0:
        nrounds = nrounds + 1
        # flip coins and count the number of heads
        heads = np.random.rand(3) < p
        tails = np.array([not i for i in heads])
        nheads = heads.sum()
    
        if nheads == 1:
            # The two tails guys give a coin to the head guy
            coins[heads] = coins[heads] + 2
            coins[tails] = coins[tails] - 1
        if nheads == 2:
            # The two head guys give a coin to the tail guy
            coins[heads] = coins[heads] - 1
            coins[tails] = coins[tails] + 2

    return nrounds

for i in xrange(ngames):
    rounds[i] = play()

#print rounds
print("Average number of rounds until game over : %.2f +- %.2f." %
      (np.average(rounds), np.std(rounds)))
