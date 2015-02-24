from itertools import chain
from collections import Counter
from math import erf, sqrt


def rank(values):
    values = list(values)
    c = Counter(values)
    ranks = {value: i+1 for i, value in enumerate(sorted(values, reverse=True))}
    for value, count in ((value, count) for value, count in c.items() if count > 1):
        ranks[value] = sum(range(ranks[value], ranks[value] - count, -1))/count

    return ranks


def z_to_p(z):
        return 0.5 * (1 + erf(z / sqrt(2)))


def calculate_z(ranks1, ranks2):
    n1, n2 = len(ranks1), len(ranks2)
    R1, R2 = sum(ranks1), sum(ranks2)
    U1 = n1 * n2 + (n1 * (n1 + 1))/2.0 - R1
    U2 = n1 * n2 + (n2 * (n2 + 1))/2.0 - R2

    assert U1 + U2 == n1 * n2

    mean = (n1*n2)/2
    n = n1+n2
    count = Counter(chain(ranks1, ranks2))
    sigma_term = sum(((value**3-value)/12 for value in count.values() if value > 1))
    std = sqrt((n1*n2/(n*(n-1)))) * sqrt(((n**3-n)/12) - sigma_term)

    U = min([U1, U2])
    z = (U-mean)/std

    return z


def mannwhitneyu(x1, x2, p=0.05):
    """
    Perform a 2-tailed Mann-Whitney U (rank sum) test.

    :param x1: Values in first sample
    :type x1: list
    :param x2: Values in second sample
    :type x2: list
    :param p: p-value against which to test (default 0.05)
    :type p: float
    :return: boolean of whether result is statistically significant at confidence level p, and the actual p-value
    :rtype: tuple
    """
    rank_dict = rank(chain(x1, x2))
    x1_ranks = [rank_dict[el] for el in x1]
    x2_ranks = [rank_dict[el] for el in x2]
    z = calculate_z(x1_ranks, x2_ranks)
    p_val = z_to_p(z)
    return p_val < p/2, p_val


if __name__ == '__main__':
    from random import random, seed
    from matplotlib import pyplot as plt
    results = []
    for delta in (x/100 for x in range(100)):
        seed()
        count = 0
        for _ in range(200):
            x1 = [random() for _ in range(100)]
            x2 = [random() + delta for _ in range(100)]
            ret, p = mannwhitneyu(x1, x2, p=0.01)
            if ret:
                count += 1
            #print(ret, p)
        results.append(count)

    plt.plot(results)
    plt.show()