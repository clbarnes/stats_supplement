import numpy as np


def _bonferroni(p_values, alpha):
    """
    Perform the `Bonferroni correction`_ on a set of p-values and test for significance.

    :param p_values: p-values to adjust for family-wise error rate
    :type p_values: sequence
    :param alpha: Confidence level between 0 and 1
    :type alpha: float
    :return: Whether or not each result was significant, and corrected p-values.
    :rtype: (numpy.array[bool], numpy.array[float])

    .. _`Bonferroni correction`: http://en.wikipedia.org/wiki/Bonferroni_correction
    """
    n = len(p_values)
    corrected_pvals = p_values*n

    return corrected_pvals < alpha, corrected_pvals


def _sidak(p_values, alpha):
    """
    Perform the `Sidak correction`_ on a set of p-values and test for significance.

    :param p_values: p-values to adjust for family-wise error rate
    :type p_values: sequence
    :param alpha: Confidence level between 0 and 1
    :type alpha: float
    :return: Whether or not each result was significant, and corrected p-values.
    :rtype: (numpy.array[bool], numpy.array[float])

    .. _`Sidak correction`: http://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction
    """
    n = len(p_values)

    if n != 0:
        correction = alpha * 1. / (1 - (1 - alpha) ** (1. / n))
    else:
        correction = 1

    corrected_pvals = p_values*correction

    return corrected_pvals < alpha, corrected_pvals


def _holm_bonferroni(p_values, alpha):
    """
    Perform the `Holm-Bonferroni correction`_ on a set of p-values and test for significance.

    :param p_values: p-values to adjust for family-wise error rate
    :type p_values: sequence
    :param alpha: Confidence level between 0 and 1
    :type alpha: float
    :return: Whether or not each result was significant, and corrected p-values. \
        Corrected p-value < alpha does not necessarily imply a significant result.
    :rtype: (numpy.array[bool], numpy.array[float])

    .. _`Holm-Bonferroni correction`: http://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    """
    sorted_pvals_tup = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected_pvals_tup = [(ind_p[0], ind_p[1]*factor) for ind_p, factor in
                           zip(sorted_pvals_tup, range(len(p_values), 0, -1))]

    trues = []
    for orig_ind, corrected_pval in corrected_pvals_tup:
        if corrected_pval < alpha:
            trues.append(orig_ind)
        else:
            break

    corrected_pvals = np.fromiter((val for _, val in sorted(corrected_pvals_tup, key=lambda x: x[0])), dtype=float)
    results = np.zeros(len(p_values), dtype=bool)
    results[trues] = True

    return results, corrected_pvals


def test_corrected(p_values, alpha=0.05, routine="holm_bonferroni"):
    """
    Correct p-values for family-wise error rate and test for significance.

    :param p_values: p-values to adjust for family-wise error rate
    :type p_values: sequence
    :param alpha: Confidence level between 0 and 1, default 0.05
    :type alpha: float
    :param routine: Which correction routine to use (default "holm_bonferroni", can also be "bonferroni" or "sidak")
    :type routine: string
    :return: Whether or not each result was significant, and corrected p-values. \
        N.B. In the case of the Holm-Bonferroni correction, corrected p-value < alpha does not necessarily imply a significant result.
    :rtype: (numpy.array[bool], numpy.array[float])
    """
    routines = {
        "holm_bonferroni": _holm_bonferroni,
        "bonferroni": _bonferroni,
        "sidak": _sidak
    }

    assert routine in routines

    pvals = np.array(p_values)

    return routines[routine](pvals, alpha)