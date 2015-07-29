import numpy as np

judges = [0.95, 0.95, 0.90, 0.90, 0.80]

def simulate_deliberation(n, same=False):

    results = np.empty(n, dtype=int)

    for i in xrange(n):
        # get votes
        r = np.random.rand(5)
        votes = r < judges
        if same:
            votes[4] = votes[0]
        vote = np.sum(votes) 
        if vote > 2:
            results[i] = 1
        else:
            results[i] = 0

    return results


if __name__ == '__main__':
    n = 100000
    np.random.seed()
    res1 = simulate_deliberation(n)
    res2 = simulate_deliberation(n, same=True)
    print np.sum(res1)/float(n), np.sum(res2)/float(n)
