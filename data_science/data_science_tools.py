#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#
# FRONT MATTER ----------------------------------------------------------------
#
###############################################################################

"""
Author:
    mtroyer

Date:
    Start: 20161118
    End: float('inf')

Purpose:
    A bunch of data management, cleaning, analysis, and visualization stuff..

Comments:
    credit is due to Joel Grus and his awesome book _Data Science from Scratch_
    for most of these..

TODO:
    chi-squared cdf - write own
    make a better comparative normal writer - accecpt mu_1, sig_1 ...

"""

###############################################################################
#
# IMPORTS ---------------------------------------------------------------------
#
###############################################################################

from __future__ import division  # Integer division is lame - use // instead
#from bs4 import BeautifulSoup
from collections import Counter
from collections import defaultdict
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
#import copy
#import csv
#import datetime
from functools import partial
#import getpass
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import operator
import os
import pandas as pd
#from pprint import pprint
import random
import re
#import requests
import sklearn
import scipy
#import statsmodels
#import sys
#import textwrap
#import traceback

###############################################################################
# Settings
###############################################################################

# Matplotlib pyplot font settings
font = {'family': 'Bitstream Vera Sans', 'weight': 'normal', 'size': 10}
matplotlib.rc('font', **font)

# a cycler for plotter colors
cycol = itertools.cycle('bgrcmk').next

#pylab
# to plot inline use __ %matplotlib inline __ at the terminal

###############################################################################
#
# DATA      -------------------------------------------------------------------
# SCIENCE   -------------------------------------------------------------------
# TOOLS     -------------------------------------------------------------------
#
###############################################################################


###############################################################################
# Central Tendency and Dispersion
###############################################################################

def mean(x):
    """Arithmetic mean"""
    return sum(x) / len(x)

def geometric_mean(x):
    return (reduce(operator.mul, x, 1))**(1/len(x))

def harmonic_mean(x):
    return len(x) / sum([1/x_ for x_ in x])

def median(v):
    """finds the 'middle-most' value of v"""
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2

    if n % 2 == 1:
        # if odd, return the middle value
        return sorted_v[midpoint]
    else:
        # if even, return the average of the middle values
        lo = midpoint - 1
        hi = midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2

def quantile(x, p):
    """returns the pth-percentile value in x"""
    p_index = int(p * len(x))
    return sorted(x)[p_index]

def mode(x):
    """returns a list, might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    mode_list = [(x_i, count) for x_i, count in counts.iteritems()
                 if count == max_count]
    if len(mode_list) == len(x):
        return 'No duplicate values'
    else:
        return mode_list

def data_range(x):  # 'range' is already taken
    return max(x) - min(x)

def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)

def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    """assumes x has at least two elements"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))

###############################################################################
# Counting Principles
###############################################################################

def permutation(n, r):
    """n combinations taken r at a time - order is important"""
    return math.factorial(n) / math.factorial(n-r)

def combination(n, r):
    """n combinations taken r at a time - order is not important"""
    return math.factorial(n) / (math.factorial(n-r) * math.factorial(r))

###############################################################################
# Correlation
###############################################################################

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero

def correlation_matrix(data):
    """returns the num_columns x num_columns matrix whose (i, j)th entry
    is the correlation between columns i and j of data"""

    _, num_columns = shape(data)

    def matrix_entry(i, j):
        return correlation(get_column(data, i), get_column(data, j))

    return make_matrix(num_columns, num_columns, matrix_entry)


###############################################################################
# Distributions
###############################################################################

def discrete_prob_dist(xs, ps):  # xs are counts ps are probs for x_i interval
    mu = sum(x_i * p_i for x_i, p_i in zip(xs, ps))
    sigma = math.sqrt(sum((x_i - mu)**2 * p_i for x_i, p_i in zip(xs, ps)))
    return mu, sigma

def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x):
    "returns the probability that a uniform random variable is less than x"
    if x < 0:   return 0    # uniform random is never less than 0
    elif x < 1: return x    # e.g. P(X < 0.4) = 0.4
    else:       return 1    # uniform random is always less than 1

def normal_pdf(x, mu=0, sigma=1):
    """the normal probabiity density function"""
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) /
            (sqrt_two_pi * sigma))

def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""

    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    # normal_cdf(-10) is ~0, normal_cdf(10)  is ~1
    low_z, hi_z = -10.0, 10.0
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2   # consider the midpoint
        mid_p = normal_cdf(mid_z)    # and the cdf's value there
        if mid_p < p:
            # midpoint is still too low, search above it
            low_z = mid_z
        elif mid_p > p:
            # midpoint is still too high, search below it
            hi_z = mid_z
        else:
            break
    return hi_z

def random_normal(mu=0, sigma=1):
    """returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random(), mu, sigma)

def binomial_pdf(r, n, p):
    return ((math.factorial(n) / (math.factorial(n-r) * math.factorial(r)))
             * p**r * (1-p)**(n-r))

def binomial_cdf(r, n, p):
    return sum([binomial_pdf(r_, n, p) for r_ in range(r+1)])

def poisson_pdf(x, mu):  # prob of x occurences with an average of mu occ.
    return (mu**x) * (math.e**-mu) / math.factorial(x)

def poisson_cdf(x, mu):
    return sum([poisson_pdf(x_, mu) for x_ in range(0, x+1)])

def bernoulli_trial(p):
    return 1 if random.random() < p else 0

def binomial_exp(p, n):
    return sum(bernoulli_trial(p) for _ in range(n))

def normal_approximation_to_binomial(n, p):
    """finds mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

###############################################################################
# HYPOTHESIS TESTING - Probabilities within an interval
###############################################################################

# the normal cdf _is_ the probability the variable is below a threshold
normal_probability_below = normal_cdf

# it's above the threshold if it's not below the threshold
def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)

# it's between if it's less than hi, but not less than lo
def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# it's outside if it's not between
def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)

###############################################################################
# HYPOTHESIS TESTING - Normal bounds
###############################################################################

def normal_upper_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """returns the symmetric (about the mean) bounds
    that contain the specified probability"""
    tail_probability = (1 - probability) / 2

    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        # if x is greater than the mean, the tail is above x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # if x is less than the mean, the tail is below x
        return 2 * normal_probability_below(x, mu, sigma)

###############################################################################
# HYPOTHESIS TESTING - A/B Test
###############################################################################

def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A, n_A, N_B, n_B, tail = 2):
    if not tail in [1,2]:
        return "tail = 1 or 2"
    else:
        p_A, sigma_A = estimated_parameters(N_A, n_A)
        p_B, sigma_B = estimated_parameters(N_B, n_B)
        test_stat = (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)
        if tail == 1:
            return test_stat, (two_sided_p_value(test_stat)/2)
        else:
            return test_stat, two_sided_p_value(test_stat)

###############################################################################
# HYPOTHESIS TESTING - z-test
###############################################################################

def two_sample_independent_z(mu1, sigma1, n1, mu2, sigma2, n2, exp_diff=0):
    """Assumes equal sample sizes and equal variances"""
    # if exp_diff, get mus in right order
    if mu1 >= mu2:
        mean_difference = mu1 - mu2 - exp_diff
    else:
        mean_difference = mu2 - mu1 - exp_diff

    pooled_stdev = math.sqrt(((sigma1**2) / n1) + ((sigma2**2) / n2))
    z_score = mean_difference / pooled_stdev
    p_value = two_sided_p_value(z_score)
    return z_score, p_value

###############################################################################
# HYPOTHESIS TESTING - t-test
###############################################################################

def two_sample_independent_t(mu1, sigma1, n1, mu2, sigma2, n2, exp_diff=0):
    """Assumes equal sample sizes and equal variances
       Equivocal to two_sample_independent_z with the exception
       that it also calculates and returns the degrees of freedom for
       for identification of the critical t-value comparison"""
    # if exp_diff, get mus in right order
    if mu1 >= mu2:
        mean_difference = mu1 - mu2 - exp_diff
    else:
        mean_difference = mu2 - mu1 - exp_diff

    pooled_stdev = math.sqrt(((sigma1**2) / n1) + ((sigma2**2) / n2))
    t_score = mean_difference / pooled_stdev
    #dof = min(n1-1, n2-1)
    dof = n1+n2-2
    p_value = stats.t.sf(np.abs(t_score), dof) * 2

    return t_score, dof, p_value

def scipy_t_test(group_1, group_2, eq_var=True):
    t, tpvalue = scipy.stats.ttest_ind(group_1, group_2, equal_var=eq_var)
    return t, tpvalue

def benjamini_hochberg(p_vals, false_disc_rate):  # list
    p_vals_ = sorted(p_vals)
    m = len(p_vals_)
    factor = false_disc_rate / m
    comp = [(p, (i + 1) * factor) for i, p in enumerate(p_vals_)]
    results = [(p, q, p < q) for p, q in comp]
    return results

#TODO: two_sample_t for unequal sample size, equal variance
#      and unequal samples size and variance - Welch's t-test

###############################################################################
# HYPOTHESIS TESTING - Baysian inference
###############################################################################

def B(alpha, beta):
    """a normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:          # no weight outside of [0, 1]
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)

###############################################################################
# HYPOTHESIS TESTING - Other tests
###############################################################################

def stat_power_test(mu1, std1, n1, mu2, std2, n2, alpha):  
    """ TODO: allow comparison to theoretical dist"""
    
    def plot_normal(mu, sigma, lbl="Sample", color='r'):
        xs = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
        ysp = [normal_pdf(xp, mu, sigma) for xp in xs]

        plt.plot(
            xs, ysp, color=color, label=('{}: {} +/-{:.3}'.format(lbl, mu, sigma)))

        plt.legend(loc='best')
        return xs
    
    # get standard errors
    sterr1, sterr2 = std1 / np.sqrt(n1), std2 / np.sqrt(n2)
    
    # get lo and hi bounds for dist1
    lo, hi = normal_two_sided_bounds((1 - alpha), mu1, sterr1)
    
    # plot them both
    s1_xs = plot_normal(mu1, sterr1, 'sample 1', color='r')
    s2_xs = plot_normal(mu2, sterr2, 'sample 2', color='b')
    
    ax = plt.gca()
    h = ax.get_ylim()[1]
    # plt.ylim([0, h * 1.5])
    ax.set_ylim([0, h * 1.5])
    ax.set_xlim([(min(min(s1_xs), min(s2_xs))),
                 (max(max(s1_xs), max(s2_xs)))])
    
    #plt.xlim([(min(min(s1_xs), min(s2_xs))),
    #          (max(max(s1_xs), max(s2_xs)))])

    # add the vertical bounds to the plot
    plt.axvline(x=lo, color='r')
    plt.axvline(x=hi, color='r')
    
    # Fill bounds
    #FN
    if mu2 - mu1 > 0:  # Mu2 is bigger - need lower boundary
        lo_xs = [x for x in s2_xs if x <= hi]
        lo_ys = [normal_pdf(lo_x, mu2, sterr2) for lo_x in lo_xs]
        plt.fill_between(lo_xs, lo_ys, color='b')
    
    if mu2 - mu1 < 0:  # Mu2 is smaller - need upper boundary
        hi_xs = [x for x in s2_xs if x >= lo]
        hi_ys = [normal_pdf(hi_x, mu2, sterr2) for hi_x in hi_xs]
        plt.fill_between(hi_xs, hi_ys, color='b')
        
    # calc beta and power
    beta = normal_probability_between(lo, hi, mu2, sterr2)
    power = 1 - beta
    print 'Alpha:\t{:.3}\nBeta:\t{}\nPower:\t{}'.format(
              alpha, beta, power)
    
    
def f_test(mu_1, sigma_1, mu_2, sigma_2):
    """Performs the standard F-test for homogeneous variances"""
    max_sigma = max(sigma_1**2, sigma_2**2)
    min_sigma = min(sigma_1**2, sigma_2**2)
    return max_sigma/min_sigma

def one_way_annova(list_of_lists):
    f, fpvalue = scipy.stats.f_oneway(list_of_lists)
    print "f: {0:.2f}\tp-value: {0:.3f}".format(f, fpvalue)


def cohens_d(mu_1, sigma_1, mu_2, sigma_2):
    """Performs Cohen's d for effect size for difference between two means
       Assumes very large homogeneous sample sizes and doesn't weight
       pooled sigmas"""
    cohens_d =  abs(mu_1 - mu_2)/math.sqrt((sigma_1**2 + sigma_2**2) / 2)
    if cohens_d <= 0.01:
        effect_size = 'Very Small'
    elif cohens_d <= 0.2:
        effect_size = 'Small'
    elif cohens_d <= 0.5:
        effect_size = 'Medium'
    elif cohens_d <= 0.8:
        effect_size = 'Large'
    else:
        effect_size = 'Very Large'

    #print "cohen's d: {}\neffect size: {}".format(cohens_d, effect_size)
    return (cohens_d, effect_size)

def chi_square(array_1, array_2):  # Make sure these are sorted!
    return sum([((o_n - e_n) ** 2) / e_n
                  for o_n, e_n in zip(array_1, array_2)])

def chi_distribution(x, df):
    #return 1 / (2 * math.gamma(df/2)) * (x/2)**(df/2-1) * math.exp(-x/2)
    return 1 - stats.chi2.cdf(x, df)

###############################################################################
# Vectors
###############################################################################

def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
   return math.sqrt(squared_distance(v, w))

###############################################################################
# Matrices
###############################################################################

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A[0].any() else 0
    return num_rows, num_cols

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]

def is_diagonal(i, j):
    """1's on the 'diagonal', 0's everywhere else"""
    return 1 if i == j else 0

def matrix_add(A, B):
    if shape(A) != shape(B):
        raise ArithmeticError("cannot add matrices with different shapes")

    num_rows, num_cols = shape(A)
    def entry_fn(i, j): return A[i][j] + B[i][j]

    return make_matrix(num_rows, num_cols, entry_fn)

###############################################################################
# Gradient Descent
###############################################################################

def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

def derivative_estimate(function, x, h=0.00001):
    # Calculate mean of h above and below for extra accuracy!
    dq_high = difference_quotient(function, x, h)
    dq_low = difference_quotient(function, x, -h)
    return (dq_high + dq_low) / 2

def partial_difference_quotient(f, v, i, h):
    # add h to just the i-th element of v
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]

def step(v, direction, step_size):
    """move step_size in the direction from v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def safe(f): # Safe meaning will give default value on f error
    """define a new function that wraps f and return it"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')         # this means "infinity" in Python
    return safe_f

# Minimization/maximazation - batch

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0                           # set theta to initial value
    target_fn = safe(target_fn)               # safe version of target_fn
    value = target_fn(theta)                  # value we're minimizing

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def negate(f):
    """return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)

# Minimization/maximazation - stochasticity

def in_random_order(data):
    """generator that returns the elements of data in random order"""
    indexes = [i for i, _ in enumerate(data)]  # create a list of indexes
    random.shuffle(indexes)                    # shuffle them
    for i in indexes:                          # return the data in that order
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    data = zip(x, y)
    theta = theta_0                             # initial guess
    alpha = alpha_0                             # initial step size
    min_theta, min_value = None, float("inf")   # the minimum so far
    iterations_with_no_improvement = 0

    # if we ever go 100 iterations with no improvement, stop
    while iterations_with_no_improvement < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # and take a gradient step for each of the data points
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn),
                               negate_all(gradient_fn),
                               x, y, theta_0, alpha_0)

###############################################################################
# Dimensionality Reduction
###############################################################################

def de_mean_matrix(A):
    """returns the result of subtracting from every value in A the mean
    value of its column. the resulting matrix has mean 0 in every column"""
    nr, nc = shape(A)
    column_means, _ = scale(A)
    return make_matrix(nr, nc, lambda i, j: A[i][j] - column_means[j])

def direction(w):
    mag = magnitude(w)
    return [w_i / mag for w_i in w]

def directional_variance_i(x_i, w):
    """the variance of the row x_i in the direction w"""
    return dot(x_i, direction(w)) ** 2

def directional_variance(X, w):
    """the variance of the data in the direction w"""
    return sum(directional_variance_i(x_i, w) for x_i in X)

def directional_variance_gradient_i(x_i, w):
    """the contribution of row x_i to the gradient of
    the direction-w variance"""
    projection_length = dot(x_i, direction(w))
    return [2 * projection_length * x_ij for x_ij in x_i]

def directional_variance_gradient(X, w):
    return vector_sum(directional_variance_gradient_i(x_i,w) for x_i in X)

def first_principal_component(X):
    guess = [1 for _ in X[0]]
    unscaled_maximizer = maximize_batch(
        partial(directional_variance, X),           # is now a function of w
        partial(directional_variance_gradient, X),  # is now a function of w
        guess)
    return direction(unscaled_maximizer)

def first_principal_component_sgd(X):
    guess = [1 for _ in X[0]]
    unscaled_maximizer = maximize_stochastic(
        lambda x, _, w: directional_variance_i(x, w),
        lambda x, _, w: directional_variance_gradient_i(x, w),
        X, [None for _ in X], guess)
    return direction(unscaled_maximizer)

def project(v, w):
    """return the projection of v onto w"""
    coefficient = dot(v, w)
    return scalar_multiply(coefficient, w)

def remove_projection_from_vector(v, w):
    """projects v onto w and subtracts the result from v"""
    return vector_subtract(v, project(v, w))

def remove_projection(X, w):
    """for each row of X
    projects the row onto w, and subtracts the result from the row"""
    return [remove_projection_from_vector(x_i, w) for x_i in X]

def principal_component_analysis(X, num_components):
    components = []
    for _ in range(num_components):
        component = first_principal_component(X)
        components.append(component)
        X = remove_projection(X, component)

    return components

def transform_vector(v, components):
    return [dot(v, w) for w in components]

def transform(X, components):
    return [transform_vector(x_i, components) for x_i in X]

###############################################################################
# Machine Learning - general
###############################################################################

# Partition datasts into train and test sets
def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):  # where x and y are realted lists of i/o
    data = zip(x, y)                   # variables that must group together
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test

# Masures of correctness- [true positive, false positive, false negative..]
def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)

def assess_model_performance(tp, fp, fn, tn):
    accuracy_ = accuracy(tp, fp, fn, tn)
    precision_ = precision(tp, fp, fn, tn)
    recall_ = recall(tp, fp, fn, tn)
    f1_ = f1_score(tp, fp, fn, tn)
    return {'accuracy  ([tp+tn]/all)': accuracy_,
            'precision (tp/[tp+tn])': precision_,
            'recall    (tp/[tp+fn])': recall_,
            'f1-score  (2pr/[p+r])': f1_}

###############################################################################
# Machine Learning - k-Nearest Neighbors
###############################################################################

def neighbor_labels(labels):
    """assumes that labels are ordered from nearest to farthest"""
    label_counts = Counter(labels)
    label_, count_ = label_counts.most_common(1)[0]
    num_labels = len([count
                       for count in label_counts.values()
                       if count == count_])

    if num_labels == 1:
        return label_                     # unique label, so return it
    else:
        return neighbor_labels(labels[:-1]) # try again without the farthest

def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda (point, _): distance(point, new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return neighbor_labels(k_nearest_labels)

def find_self_knns(labeled_points):
    """each labeled point should be a pair (point, label)"""
    results_list = []
    for k_ in [1,3,5,7, 9]:
        num_correct = 0
        for item in labeled_points:
            point, label = item
            other_items = [item_ for item_ in labeled_points if item_ != item]
            prediction_ = knn_classify(k_, other_items, point)
            if prediction_ == label:
                num_correct += 1
        result_ = (k_, num_correct/len(labeled_points)*100)
        results_list.append(result_)
    for res in results_list:
        print res[0], "neighbors[s]:", str(res[1])[:4], "percent correct"
    #plt.bar([k-0.4 for k, _ in results_list], [pc for _, pc in results_list])
    return results_list

def plot_2d_knn_training_data(labeled_points, lbl_list):
    """each labeled point should be a pair (2d-point, label)"""
    plots = {label : ([],[]) for label in lbl_list}
    markers = {key : random.choice("*+ox.") for key in lbl_list}
    colors = {key : random.choice("bgrcmyk") for key in lbl_list}
    for (x, y), label_ in labeled_points:
        plots[label_][0].append(x)
        plots[label_][1].append(y)
    for lbl, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[lbl], marker=markers[lbl], label=lbl)
    plt.legend(loc=0)

def plot_knn_area(k, labeled_points):
    """each labeled point should be a pair (2d-point, label)"""
    plots = {label : ([],[]) for _, label in labeled_points}
    xmin = min(point[0] for point, _ in labeled_points)
    xmax = max(point[0] for point, _ in labeled_points)
    ymin = min(point[1] for point, _ in labeled_points)
    ymax = max(point[1] for point, _ in labeled_points)
    xs = [xmin + n * ((xmax-xmin)/100.0) for n in range(101)]
    ys = [ymin + n * ((ymax-ymin)/100.0) for n in range(101)]
    for x in xs:
        for y in ys:
            prediction = knn_classify(k, labeled_points, [x,y])
            plots[prediction][0].append(x)
            plots[prediction][1].append(y)
    for lbl, (x, y) in plots.items():
        plt.scatter(x, y, color=cycol(), label=lbl)
    plt.legend(loc=0)

###############################################################################
# Simple Linear Regression
###############################################################################

def predict(alpha, beta, x_i):
    return beta * x_i + alpha

def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))

def least_squares_fit(x,y):
    """given training values for x and y,
    find the least-squares values of alpha and beta"""
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha, beta, x, y):
    """the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model"""

    return 1.0 - (sum_of_squared_errors(alpha, beta, x, y) /
                  total_sum_of_squares(y))

def squared_error(x_i, y_i, theta):
    alpha, beta = theta
    return error(alpha, beta, x_i, y_i) ** 2

def squared_error_gradient(x_i, y_i, theta):
    alpha, beta = theta
    return [-2 * error(alpha, beta, x_i, y_i),       # alpha partial derivative
            -2 * error(alpha, beta, x_i, y_i) * x_i] # beta partial derivative


###############################################################################
# Search Functions
###############################################################################

def regex_walk(regex, dir_path):
    """ recursievely walk a dir and execute a regex search on
    acceptable file contents"""
    match_list = []
    if not os.path.exists(dir_path):
        print "Directory does not exist or is not supported"
        return None
    try:
        ro = re.compile(regex) # may need "r'" "'"
    except:
        print "Incorrect format for regex"
        return None

    ext_list = ['.txt', '.py', '.pyt', '.pyw']

    def regex_search(dir_, file_, ro):
        rs_matches = []
        txtfile = open(os.path.join(dir_, file_))
        txtlines = txtfile.readlines()
        for line_ in txtlines:
            mo = ro.findall(line_)
            if mo:
                rs_matches.append([os.path.join(dir_, file_), mo,
                                   str(line_).strip().strip("\n")])

                print "REGEX MATCH\nFile:\n{}\nMatch:\n{}\nLine:\n{}\n"\
                      "".format(file_, mo, line_)
        return rs_matches

    for dir_, folders, files in os.walk(dir_path):
        print "\nRecursively searching the following files "\
              "and folders for r'{}'\n".format(regex)

        print str(dir_)
        for folder in folders:
            print str(folder)
        for file_ in files:
            print str(file_)
        print "\n"
        for file_ in files:
            _, ext = os.path.splitext(file_)
            if ext in ext_list:
                match_list.extend(regex_search(dir_, file_, ro))
        for folder_ in folders:
            match_list.extend(regex_walk(regex,
                                         os.path.join(dir_, folder_)))
    if not match_list:
        print "No Matches"
        return []
    else:
        return match_list

###############################################################################
# Text Processing
###############################################################################

def count_words(txt):
    # Parse the txt and create a dictionary of word counts
    word_dict = defaultdict(int)
    with open(txt, 'r') as textfile:
        for line in textfile.readlines():
            clean_line = re.sub('[^a-zA-Z0-9_ ]', '', line)
            for word in clean_line.split():
                word_dict[word.lower()] += 1
    return word_dict

def counts_dict_add_prop(dict_):
    n = sum(dict_.values())
    dd = defaultdict(list)
    for k, v in dict_.items():
        dd['word'].append(k)
        dd['count'].append(v)
        dd['proportion'].append(v/n)
        dd['length'].append(len(k))

    dataframe = pd.DataFrame(dd)
    return dataframe

def plot_word_length(path, dataframe, x_label='Length',
                                      y_label='Proportion',
                                      title='Word Length and Proportion'):

    word_len_and_prop = dataframe.groupby('length').sum()
    plt.plot(word_len_and_prop.index, word_len_and_prop['proportion'],
             label='file: {}'.format(os.path.basename(path)))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

def count_words_and_plot(txt):
    counted_words = count_words(txt)
    word_count_df = counts_dict_add_prop(counted_words)
    plot_word_length(txt, word_count_df)



def word_freq_processor(compare_words, comparison_texts):
    #

    stemmer = nltk.stem.porter.PorterStemmer()  # Does the good good work

    def stem_tokens(tokens, stemmer):
        sts = [stemmer.stem(item) for item in tokens]
        return sts

    def tokenize(text):
        return stem_tokens(nltk.word_tokenize(text), stemmer)

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
                                            tokenizer=tokenize,
                                            stop_words='english')

    vectored = vectorizer.fit(compare_words)
    targets = [vectored.transform(item) for item in comparison_texts]
    print vectored.get_feature_names()
    target_arrays = [item.toarray() for item in targets]
    return target_arrays

###############################################################################
# Naive Bayes text processing
###############################################################################

def tokenize(message):
    message = message.lower()                       # convert to lowercase
    all_words = re.findall("[a-z0-9']+", message)   # extract the words
    return set(all_words)

def count_training_words(training_set):
    """training set consists of pairs (message, flag)"""
    counts = defaultdict(lambda: [0, 0])
    for message, flag in training_set:
        for word in tokenize(message):
            counts[word][0 if flag else 1] += 1
    return counts

def word_probabilities(counts, total_flagged, total_unflagged, k=0.5):
    """turn the word_counts into a list of triplets
    w, p(w | flag) and p(w | ~flag)"""
    return [(w,
             (flag + k) / (total_flagged + 2 * k),
             (non_flag + k) / (total_unflagged + 2 * k))
             for w, (flag, non_flag) in counts.iteritems()]

def flag_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_flag = log_prob_if_not_flag = 0.0

    for word, prob_if_flag, prob_if_not_flag in word_probs:

        # for each word in the message,
        # add the log probability of seeing it
        # logs are more float friendly than repeatedly multiplying probs
        if word in message_words:
            log_prob_if_flag += math.log(prob_if_flag)
            log_prob_if_not_flag += math.log(prob_if_not_flag)

        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_flag += math.log(1.0 - prob_if_flag)
            log_prob_if_not_flag += math.log(1.0 - prob_if_not_flag)

    prob_if_flag = math.exp(log_prob_if_flag)
    prob_if_not_flag = math.exp(log_prob_if_not_flag)
    return prob_if_flag / (prob_if_flag + prob_if_not_flag)

class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):

        num_flags = len([flag
                         for message, flag in training_set
                         if flag])
        num_non_flags = len(training_set) - num_flags

        # run training data through our "pipeline"
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_flags,
                                             num_non_flags,
                                             self.k)

    def classify(self, message):
        return flag_probability(self.word_probs, message)


def train_and_test_model(data):
    """Data is list [text, flag]"""
    train_data, test_data = split_data(data, 0.75)

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    classified = [(text, flag, classifier.classify(text))
              for text, flag in test_data]

    counts = Counter((is_flag, flag_probability > 0.5) # TF (actual, predicted)
                     for _, is_flag, flag_probability in classified)
    return counts

#TODO:
#min count to flag
#stem search words - Porter Stemmer
#Improve/expand tokenizer

###############################################################################
# Working with functions
###############################################################################

#TODO - make more general
def try_or_none(f):
    """wraps f to return None if f raises an exception
    assumes f takes only one input"""
    def f_or_none(x):
        try: return f(x)
        except: return None
    return f_or_none

###############################################################################
# Working with data
###############################################################################

def bucketize(point, bucket_size):
    """floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)

def bucketize_points(points, bucket_size):
    """buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)

def parse_rows_with(reader, parsers):
    """wrap a reader to apply the parsers to each of its rows"""
    for row in reader:
        yield parse_row(row, parsers)

def parse_row(input_row, parsers):
    return [try_or_none(parser)(value) if parser is not None else value
            for value, parser in zip(input_row, parsers)]

def try_parse_field(field_name, value, parser_dict):
    """try to parse value using the appropriate function from parser_dict"""
    parser = parser_dict.get(field_name) # None if no such entry
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value

def parse_dict(input_dict, parser_dict):
    return { field_name : try_parse_field(field_name, value, parser_dict)
             for field_name, value in input_dict.iteritems() }

def picker(field_name):
    """"returns a function that picks a field out of a dictionary"""
    return lambda row: row[field_name]

def pluck(field_name, list_dict):
    return map(picker(field_name), list_dict)

def group_by(grouper, list_dict):
    grouped = defaultdict(list)
    for row in list_dict:
        grouped[grouper(row)].append(row)
    return grouped

def scale(data_matrix):
    num_rows, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix,j))
             for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix,j))
              for j in range(num_cols)]
    return means, stdevs

def rescale(data_matrix):
    """rescales the input data so that each column
    has mean 0 and standard deviation 1
    ignores columns with no deviation"""
    means, stdevs = scale(data_matrix)

    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]

    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescaled)

def describe_data(data, name=''):  # one-dimensional [at least one at a time]
    # Check for types
    n_ = len(data)
    min_ = min(data)
    max_ = max(data)
    range_ = max_ - min_
    mean_ = mean(data)
    median_ = median(data)
    mode_ =  mode(data)
    stDev = standard_deviation(data)
    CoV = stDev / mean_
    stErr_est = stDev / math.sqrt(n_)
    description = {'sample size'                : n_,
                   'minimum'                    : min_,
                   'maximum'                    : max_,
                   'range'                      : range_,
                   'mean'                       : mean_,
                   'median'                     : median_,
                   'mode'                       : mode_,
                   'standard deviation'         : stDev,
                   'coefficient of variation'   : CoV,
                   'standard error estimate'    : stErr_est}
    print
    print str(name).center(50, '=')
    print
    sort_list = ['sample size', 'minimum', 'maximum', 'range', 'mean',
                 'median', 'mode', 'standard deviation',
                 'coefficient of variation', 'standard error estimate']
    for k in sort_list:
        v = description[k]
        print k.rjust(25)+':', str(v).ljust(20)
    return description

def one_d_numeric_description(data, name=''):
    # assumes list or other simple iterable
    description = describe_data(data, name)
    test_for_normal(data)
    bucket_size = round(description['range']) / 10
    plot_bucket_histogram(data, bucket_size, name)
    return description

def n_d_numeric_description(data):
    # get the individual lists
    field_dict = defaultdict(list)
    for row in data:
        counter = 1
        for field in row:
            field_dict['variable_'+str(counter)].append(field)
            counter += 1
    for k, v in sorted(field_dict.items()):
        one_d_numeric_description(v, k)
    # assumes a list of n-dimensional lists
    print
    print str('scatterplot matrix').center(50, '=')
    make_scatterplot_matrix(data)
    print
    print str('correlation matrix').center(50, '=')
    print
    cm = [[str(round(field, 2)).rjust(5) for field in fields]
                  for fields in correlation_matrix(data)]
    for row in cm:
        print row
    return cm

def test_for_normal(data, p_value=0.05):
    kurt = scipy.stats.kurtosis(data)
    skew = scipy.stats.skew(data)
    kzscore, kpvalue = scipy.stats.kurtosistest(data)
    szscore, spvalue = scipy.stats.skewtest(data)
    print
    print "Kurtosis: {0:.3f}\nz-score: {1:.03f}\np-value: {2:.03f}"\
          "".format(kurt, kzscore, kpvalue)
    print
    print "Skew: {0:.3f}\nz-score: {1:.03f}\np-value: {2:.03f}"\
          "".format(skew, szscore, spvalue)
    print
    if kpvalue < p_value or spvalue < p_value:
        print "Data is not normally distributed"
    else: print "Data is normally distributed"
    #plot_bucket_histogram(data, 0.5)
    return ((kurt, kzscore, kpvalue),(skew, szscore, spvalue))

def transform_one_variable(group_1, group_2):
    # Build a transformation dictionary
    transformations = {'x'        : lambda x: x ,
                       '1/x'      : lambda x: 1/x,
                       'x**2'     : lambda x: x**2,
                       'x**3'     : lambda x: x**3,
                       'log(x)'   : lambda x: np.log(x),
                       'sqrt(x)'  : lambda x: math.sqrt(x),
                       'exp(x)'   : lambda x: np.exp(x),
                       'log(1/x)' : lambda x: np.log(1/x)}
    # Transform group 2 and check pearson's r for non-parametric correleation
    for trans in sorted(transformations):
        pr_c, pr_p = scipy.stats.pearsonr(group_1,
                                         [transformations[trans](x)
                                          for x in group_2])
        print ("Transformation: {}".format(trans)).ljust(25),\
              ("Pearson's r: {0:.3f}".format(pr_c)).rjust(10)

def PolynomialFeatures_labeled(input_df,power):
    poly = PolynomialFeatures(power)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(input_feature_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s^%d" % (variable,power)
                if final_label == "":         #If the final label isn't yet specified
                    final_label = intermediary_label
                else:
                    final_label = final_label + " x " + intermediary_label
        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
    return output_df

################################################################################
# 14c functions
###############################################################################

def chi2_for_equal_prop(pdf_1, mu_1, sig_1, n_1,
                        pdf_2, mu_2, sig_2, n_2,
                        num_bins=5):

    """Pearson's chi-squared goodness-of-fit test for 14c Dates"""
    dof = num_bins-1
    x_lower = min(mu_1 - 3.0 * sig_1, mu_2 - 3.0 * sig_2)  # 99% of func
    x_upper = max(mu_1 + 3.0 * sig_1, mu_2 + 3.0 * sig_2)  # 99% of func
    bin_size = (x_upper - x_lower)/num_bins
    chi_bins = [bin_size * i + x_lower for i in range(0, num_bins)]
    pdf_1_dd = defaultdict(int)
    pdf_2_dd = defaultdict(int)

    ### Sample each function / bin and return the probability over bin range
    for bin_ in chi_bins:
        lower, upper = bin_, bin_ + bin_size
        pdf_1_dd[bin_] = normal_probability_between(lower, upper, mu_1, sig_1)
        pdf_2_dd[bin_] = normal_probability_between(lower, upper, mu_2, sig_2)

    # return as arrays of PROPORTIONS
    pdf_1_array = [item[1] * n_1 for item in sorted(pdf_1_dd.items())]
    pdf_2_array = [item[1] * n_2 for item in sorted(pdf_2_dd.items())]

    chi_square_score = chi_square(pdf_1_array, pdf_2_array)
    p_value = chi_distribution(chi_square_score, dof)

    return (chi_square_score, dof, p_value)

###############################################################################
# if '__name__' == '__main__'
###############################################################################

if __name__ == "__main__":
    group_1 = [random.randint(0, 10) for i in range(200)]
    group_2 = [random.normalvariate(10,1) for i in range(200)]
    test_for_normal(group_2)
    #test_for_normal(group_2)
    #transform_one_variable(group_1, group_2)

###############################################################################
# Production
###############################################################################

#    # Read the input data into a dataframe object
#    csv_path = r'C:\Users\mtroyer\python\BCNM_Python.csv'
#    df = pd.read_csv(csv_path)
#
#    # parse the point (x,y) and label into knn
#    # each labeled point should be a pair ([x,y], label)
#
#    mat_xy = df[['Mat_Label', 'SITE_ID', 'X', 'Y']]
#    filtered_xy = mat_xy[mat_xy['SITE_ID'] == '5CF.88']
#    unique_ = list(df['Mat_Label'].unique())
#
#    data_list = [([row[3],row[4]], row[1]) for row in filtered_xy.itertuples()]
#
#    #knn
#    #plot_2d_knn_training_data(data_list, unique_)
#    find_self_knns(data_list)
#    plot_knn_area(5, data_list)
#
#    test_data_list = [[int(random_normal() *10) for i in range(4)]
#                       for i in range(1000)]
#    n_d_numeric_description(test_data_list)
#
## data ------------------------------------------------------------------------
#
#    alpha = 0.10
#
#    sample_1 = {'id'        : 'Sample 1',
#                'mean'      :       1000,
#                'sigma'     :         55,
#                'sample_n'  :         20}
#
#    sample_2 = {'id'        : 'Sample 2',
#                'mean'      :       1030,
#                'sigma'     :         50,
#                'sample_n'  :         20}
#
## process ---------------------------------------------------------------------
#
#    print
#    print 'Testing for difference between two sample means'.center(100, '=')
#    print
#
#    print '---Sample 1---'
#    print
#    for k, v in sorted(sample_1.items()):
#        print str(k).rjust(10), str(v).rjust(10)
#    print
#    print
#    print '---Sample 2---'
#    print
#    for k, v in sorted(sample_2.items()):
#        print str(k).rjust(10), str(v).rjust(10)
#    print
#    print
#
#    # f-test for homogeneous variances
#    ft = f_test(sample_1['mean'], sample_1['sigma'],
#                sample_2['mean'], sample_2['sigma'])
#    print 'f-test for homogeneous variances between two samples:'
#    print 'f-test score: {}'.format(ft)
#    print
#    print
#
#    # cohen's d for standard difference between means
#    cd, es = cohens_d(sample_1['mean'], sample_1['sigma'],
#                      sample_2['mean'], sample_2['sigma'])
#    print "Cohen's-d - the standard difference between two means - effect size"
#    print "Cohen's-d score: {}".format(cd)
#    print 'effect size: {}'.format(es)
#    print
#    print
#
#    if sample_1['sample_n'] >= 30 and sample_2['sample_n'] >= 30:
#        # z-test - both n >= 30
#        print 'z-test for difference between two independent samples'
#        print '- assumes roughly equal variances - see f-test above -'
#        z, z_p = two_sample_independent_z(sample_1['mean'],
#                                          sample_1['sigma'],
#                                          sample_1['sample_n'],
#                                          sample_2['mean'],
#                                          sample_2['sigma'],
#                                          sample_2['sample_n'])
#
#        print 'z-test score: {}\np-value: {}'.format(z, z_p)
#        if z_p < alpha:
#            print 'Reject null hypothesis'
#        else:
#            print 'Do not reject null hypothesis'
#        print
#        print
#
#    else:
#        # t-test 1 or both n < 30
#        print 't-test for difference between two independent samples'
#        print '- assumes roughly equal variances - see f-test above -'
#        t, dof, t_p = two_sample_independent_t(sample_1['mean'],
#                                               sample_1['sigma'],
#                                               sample_1['sample_n'],
#                                               sample_2['mean'],
#                                               sample_2['sigma'],
#                                               sample_2['sample_n'])
#
#        print 't-test score: {}\ndegrees of freedom: {}\np-value: {}'\
#              ''.format(t, dof, t_p)
#        if t_p < alpha:
#            print 'Reject null hypothesis'
#        else:
#            print 'Do not reject null hypothesis'
#        print
#        print
#
#    # chi-square goodness of fit test two distribution functions
#    print 'chi-square goodness of fit for two distributions [k-1]'
#    x2, dof, x2p = chi2_for_equal_prop(normal_pdf,
#                                       sample_1['mean'],
#                                       sample_1['sigma'],
#                                       sample_1['sample_n'],
#                                       normal_pdf,
#                                       sample_2['mean'],
#                                       sample_2['sigma'],
#                                       sample_2['sample_n'],
#                                       num_bins=5)
#
#    print 'chi-square test score: {}\ndegrees of freedom: {}\np-value: {}'\
#          ''.format(x2, dof, x2p)
#    if x2p < alpha:
#        print 'Reject null hypothesis'
#    else:
#        print 'Do not reject null hypothesis'
#    print
#
#    # visualize the input data
#    print 'Comparison of uncalibrated probability density functions'
#    plot_normal_pdf(sample_1['mean'], sample_1['sigma'], sample_1['id'])
#    plot_normal_pdf(sample_2['mean'], sample_2['sigma'], sample_2['id'])
