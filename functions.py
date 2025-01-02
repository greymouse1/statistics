import pandas
from scipy import stats
import numpy
import itertools
from tabulate import tabulate
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def get_confidence_interval(data, confidence=0.95):
    """ Determines the confidence interval for a given set of data, 
        assuming the population standard deviation is not known.

    Args:  # 'arguments', or inputs to the function
        data (single-column or list): The data
        confidence (float): The confidence level on which to produce the interval.

    Returns:
        c_interval (tuple): The confidence interval on the given data (lower, upper).
    """

    n = len(data)  # determines the sample size
    m = numpy.mean(data)  # obtains mean of the sample

    se = stats.sem(data)  # obtains standard error of the sample

    c_interval = stats.t.interval(confidence, n-1, m, se)  # determines the confidence interval
    return c_interval  # which is of the form (lower bound, upper bound)

def t_test(data_group1, data_group2, confidence=0.95):
    alpha = 1-confidence

    if stats.levene(data_group1, data_group2)[1]>alpha:
        equal_variance = True
    else:
        equal_variance = False

    t, p = stats.ttest_ind(data_group1, data_group2, equal_var = equal_variance)

    reject_H0 = "True"
    if p > alpha:
        reject_H0 = "False"

    return({'t': t, "p": p, "Reject H0": reject_H0})

def ANOVA(dataset,independent,dependent,confidence = 0.95):
    """ Calculates the ANOVA for a given dataset and prints an ANOVA table
        and results of post hoc test if test was performed.

    Args:  # 'arguments', or inputs to the function
        dataset (pandas.DataFrame): The data
        independent (string): The name of the independent column.
        dependent (string): The name of the dependent column.
        confidence (float): The desired confidence level for the ANOVA.

    Returns:
        None.
    """
    groups = pandas.unique(dataset[independent])
    k = len(groups)  # number of groups
    n = len(dataset[dependent])  # number of dependent data points

    # here we calculate the three degrees of freedom used in the ANOVA
    DFbetween = k - 1
    DFwithin = n - k
    DFtotal = n - 1

    # we use textbook notation:
    # x_dd = sum over i and j x_ij
    # x_id = sum over j x_ij
    # x_dj = sum over i x_ij
    # where i is the independent variable and j is the dependent variable

    x_dd = sum(dataset[dependent])
    CF = (x_dd**2)/n

    SStotal = sum(x_ij**2 for x_ij in dataset[dependent]) - CF

    SSbetween = 0
    for i in groups:
        group_data = dataset.loc[dataset[independent]==i]
        n_i = len(group_data[dependent])
        x_id = sum(group_data[dependent])
        SSbetween += (x_id**2)/n_i

    SSbetween = SSbetween - CF  # so^2 - s^2

    SSwithin = SStotal - SSbetween

    MSbetween = SSbetween/DFbetween
    MSwithin = SSwithin/DFwithin

    F = MSbetween/MSwithin
    p = stats.f.sf(F, DFbetween, DFwithin)

    print(tabulate([['Between', DFbetween, SSbetween, MSbetween, F],
                    ['Within', DFwithin, SSwithin, MSwithin, ' '],
                    ['Total', DFtotal, SStotal, ' ', ' ']],
    headers=['Variation due to', 'DoF','Sum of squares','mean squares','F ratio']))
    print('Significance (p value): '+str(p))
    print('\n')
    alpha = 1-confidence
    if p < alpha:
        print("Reject null-hypothesis: There are statistical differences present.")
        print(pairwise_tukeyhsd(dataset[dependent], dataset[independent], alpha=alpha))
    else:
        print("Fail to reject the null-hypothesis: There are no statistical differences present at this level of significance.")