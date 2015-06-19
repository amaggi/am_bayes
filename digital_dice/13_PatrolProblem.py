import numpy as np
import matplotlib.pyplot as plt

def distance_to_accident(x, y, pc_lane, ac_lane, grass=True):
    """
    pc_lane can be either 1 or 2
    ac_lane can be either 1 or 2
    direction can be either 0 or 1
    """

    direction = get_direction(x, y, pc_lane)

    if pc_lane == 1:
        if ac_lane == 1:
            if direction == 0:
                if grass:
                    d = y-x
                else:
                    d = 2+x-y
            else:
                d = x-y
        else:
            if direction == 0:
                if grass:
                    d = y-x
                else:
                    d = 2-x-y
            else:
                if grass:
                    d = x-y
                else:
                    d = 2-x-y
    else:
        if ac_lane == 1:
            if direction == 0:
                if grass:
                    d = x-y
                else:
                    d = x+y
            else:
                if grass:
                    d = y-x
                else:
                    d = x+y
        else:
            if direction == 0:
                if grass:
                    d = x-y
                else:
                    d = 2-x+y
            else:
                    d = y-x
 
    return d


def get_direction(x, y, pc_lane):
    if pc_lane == 1:
        if y <= x:
            direction = 1
        else:
            direction = 0
    else:
        if y <= x:
            direction = 0
        else:
            direction = 1

    return direction
        

def run_accidents(n_pc, grass=True):

    n = 2000
    np.random.seed()
    x = np.random.rand(n)
    y = np.random.rand(n_pc*n).reshape(n, n_pc)
    pc_lane = np.random.randint(1, 2, size=n_pc*n).reshape(n, n_pc)
    ac_lane = np.random.randint(1, 2, size=n)

    results_a = np.empty((n), dtype=float)
    results_b = np.empty((n), dtype=float)
    results_c = np.empty((n, n_pc), dtype=float)

    for i in xrange(n):
        results_a[i] = distance_to_accident(x[i], 0.5, pc_lane[i, 0],
                                            ac_lane[i], grass=grass)
        results_b[i] = distance_to_accident(x[i], y[i, 0], pc_lane[i, 0],
                                            ac_lane[i], grass=grass)
        for ic in xrange(n_pc):
            results_c[i, ic] = distance_to_accident(x[i], y[i, ic],
                                                    pc_lane[i, ic], ac_lane[i],
                                                    grass=grass)

    return np.average(results_a), np.average(results_b), \
           np.average(np.min(results_c, axis=1))


if __name__ == '__main__':

    n_pc_max = 10 

    results_grass = np.empty((n_pc_max, 3), dtype=float)
    results_concrete = np.empty((n_pc_max, 3), dtype=float)

    # check for different numbers of patrol cars
    for n_pc in xrange(n_pc_max):
        results_grass[n_pc, :] = run_accidents(n_pc+1, grass=True)
        results_concrete[n_pc, :] = run_accidents(n_pc+1, grass=False)

    plt.figure()
    cars = np.arange(n_pc_max)+1
    plt.plot(1, results_grass[0, 0], 'x', label='(a) Grass')
    plt.plot(1, results_grass[0, 1], 'o', label='(b) Grass')
    plt.plot(cars, results_grass[:, 2], label='(c) Grass')
    plt.plot(1, results_concrete[0, 0], 'x', label='(a) Concrete')
    plt.plot(1, results_concrete[0, 1], 'o', label='(b) Concrete')
    plt.plot(cars, results_concrete[:, 2], label='(c) Concrete')

    plt.legend()

    plt.xlabel('Number of patrol cars')
    plt.ylabel('Average distance to accident')
    plt.xlim([0, n_pc_max+1])

    plt.show()
