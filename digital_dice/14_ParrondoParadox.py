import numpy as np
import matplotlib.pyplot as plt


def flip_coin(p, epsilon, M):

    r = np.random.rand()
    if r < p-epsilon:
        return M+1
    else:
        return M-1


def play_A(epsilon, M):
    return flip_coin(0.5, epsilon, M)


def play_B(epsilon, M):
    if np.mod(M, 3) == 0:
        return flip_coin(0.1, epsilon, M)
    else:
        return flip_coin(0.75, epsilon, M)


def play_sequence(n, epsilon, game='AB'):

    M = np.empty(n+1, dtype=int)
    M[0] = 0

    for i in xrange(n):
        if game=='A':
            M[i+1] = play_A(epsilon, M[i])
        elif game=='B':
            M[i+1] = play_B(epsilon, M[i])
        else:
            r = np.random.rand()
            if r < 0.5:
                M[i+1] = play_A(epsilon, M[i])
            else:
                M[i+1] = play_B(epsilon, M[i])
                
    return M


def iterate_games(n_seq, seq_length, epsilon, game='AB'):

    M_out = np.empty((n_seq, seq_length+1), dtype=int)

    for i in xrange(n_seq):
        M_out[i, :] = play_sequence(seq_length, epsilon, game)

    return M_out


def plot_result(M_out, title):

    n, seq_len = M_out.shape
    M_mean = np.mean(M_out, axis=0)
    M_std = np.std(M_out, axis=0)
    M_median = np.median(M_out, axis=0)
    M_5 = np.percentile(M_out, 5, axis=0)
    M_95 = np.percentile(M_out, 95, axis=0)
    x = np.arange(seq_len)


    plt.figure(facecolor='w')
    plt.errorbar(x, M_median, yerr=[M_median-M_5, M_95-M_median], fmt='o',
                 label='median')
    plt.errorbar(x, M_mean, yerr=M_std, fmt='o', label='mean')
    plt.legend(loc='upper left')

    plt.xlabel('Throw number')
    plt.ylabel('Capital')

    plt.title(title)


if __name__ == '__main__':

    epsilon = 0.005
    np.random.seed()
    M_A = iterate_games(3000, 100, epsilon, game='A')
    M_B = iterate_games(3000, 100, epsilon, game='B')
    M_AB = iterate_games(3000, 100, epsilon, game='C')

    plot_result(M_A, 'Game A')
    plot_result(M_B, 'Game B')
    plot_result(M_AB, 'Game AB')

    plt.show()
