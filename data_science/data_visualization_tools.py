#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np
import data_science_tools as dst
import matplotlib.pyplot as plt

###############################################################################
# Visualizations
###############################################################################

def plot_normal(mu, sigma, lbl="Sample", plot_type='pdf'):
    xs = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
    ysp = [dst.normal_pdf(xp, mu, sigma) for xp in xs]
    ysc = [dst.normal_cdf(xc, mu, sigma) for xc in xs]
    
    if plot_type == 'pdf':
        plt.plot(
        xs, ysp, label=('{}: {} +/-{}'.format(lbl, mu, sigma)))
    
    if plot_type == 'cdf':
        plt.plot(
        xs, ysc, label=('{}: {:.3} +/-{:.3}'.format(lbl, mu, sigma)))
        
    ax = plt.gca()
    h = ax.get_ylim()[1]
    plt.ylim([0, h * 1.2])
        
    plt.legend()
    return

def plot_normal_w_fill(mu, sigma, alpha, lbl="Sample", color='r'):
    xs = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 200)
    ysp = [dst.normal_pdf(xp, mu, sigma) for xp in xs]
    
    plt.plot(
        xs, ysp, color=color, label=('{}: {} +/-{}'.format(lbl, mu, sigma)))

    lo, hi = dst.normal_two_sided_bounds((1 - alpha), mu, sigma)
    
    # get lower range
    lo_xs = [x for x in xs if x <= lo]
    lo_ys = [dst.normal_pdf(lo_x, mu, sigma) for lo_x in lo_xs]
    
    # get upper range
    hi_xs = [x for x in xs if x >= hi]
    hi_ys = [dst.normal_pdf(hi_x, mu, sigma) for hi_x in hi_xs]    
    
    plt.fill_between(lo_xs, lo_ys, color=color)
    plt.fill_between(hi_xs, hi_ys, color=color)

    ax = plt.gca()
    h = ax.get_ylim()[1]
    plt.ylim([0, h * 1.2])
        
    plt.legend(loc='best')
    return

def plot_binomial_exp(p, n, num_points):

    data = [dst.binomial_exp(p, n) for _ in range(num_points)]

    # use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')

    mu = p * n
    sigma = np.sqrt(n * p * (1 - p))

    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [dst.normal_cdf(i + 0.5, mu, sigma) - dst.normal_cdf(i - 0.5, mu, sigma)
          for i in xs]
    plt.plot(xs,ys)
    plt.title("Binomial Distribution vs Normal Approximation\n\
    p = %.2f, n = %i, Trials = %i" % (p, n, num_points))
    plt.show()


def make_count_hist(data_list, plot_type='frequency'):
    if not plot_type.lower() in ['frequency', 'proportion', 'both']:
        return "Incorrect plot type use 'frequency', 'proportion', 'both'"

    # get a list of counter (item, count) tuples
    hist = sorted(Counter(data_list).items())
    n_ = float(len(data_list))
    y_label = plot_type.title()
    # get the xs ys and ys proportions
    x_list = [x[0] for x in hist]
    y_list = [x[1] for x in hist]
    y_prop = [y/n_ for y in y_list]

    # Check if all numbers else treat as strings - the easy way
    try:
        sum(x_list)
        xs = [x - 0.4 for x in x_list]
        x_adj = x_ticks = None
    except:
        # prepare to plot by adjusted index position and then change labels
        xs = [str(x) for x in x_list]
        x_adj = [i + 0.4 for i, _ in enumerate(xs)]
        x_ticks = [v for _, v in enumerate(xs)]

    if plot_type.lower() == 'frequency':
        y_upper = 1.1 * max(y_list)
        if x_adj:
            plt.bar(x_adj, y_list, 0.8, color='black')
            # rename ticks and set xmax
            plt.xticks([x + 0.4 for x in x_adj], x_ticks)
            plt.xlim(xmin=0, xmax=len(xs)+0.6)
        else:
            plt.bar(xs, y_list, 0.8, color='black')
        plt.ylim(ymin=0, ymax=y_upper)
        plt.ylabel(y_label)
        plt.show()

    elif plot_type.lower() == 'proportion':
        y_upper = 1.1 * max(y_prop)
        if x_adj:
            plt.bar(x_adj, y_prop, 0.8, color='black')
            # rename ticks and set xmax
            plt.xticks([x + 0.4 for x in x_adj], x_ticks)
            plt.xlim(xmin=0, xmax=len(xs)+0.6)
        else:
            plt.bar(xs, y_prop, 0.8, color='black')
        plt.ylim(ymin=0, ymax=y_upper)
        plt.ylabel(y_label)
        plt.show()

    return

def make_scatterplot(xs, ys, labels=None, title=None,
                     xaxis=None, yaxis=None):  # xs, ys, labels -3 sorted lists
    plt.scatter(xs, ys)
    # label each point
    if labels:
        for label, x, y in zip(labels, xs, ys):
            plt.annotate(label,
                         xy=(x, y), # put the label with its point
                         xytext=(1, -1), # but slightly offset
                         textcoords='offset points')
    if title:
        plt.title(title)
    if xaxis:
        plt.xlabel(xaxis)
    if yaxis:
        plt.ylabel(yaxis)
    plt.show()

def make_pie_chart(prop_list, labels=None):
    if not labels:
        labels = range(1, len(prop_list)+1)
    if not round(sum(prop_list)) == 1.0:
        return 'The proportions do not add up to 1'
    print 'labels', labels
    print 'proportion', round(sum(prop_list))
    plt.pie(prop_list, labels=labels)
    plt.axis("equal")
    plt.show()

def make_scatterplot_matrix(data):

    _, num_columns = dst.shape(data)
    fig, ax = plt.subplots(num_columns, num_columns)

    for i in range(num_columns):
        for j in range(num_columns):

            # scatter column_j on the x-axis vs column_i on the y-axis
            if i != j: ax[i][j].scatter(dst.get_column(data, j),
                                        dst.get_column(data, i))

            # unless i == j, in which case show the series name
            else: ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                                    xycoords='axes fraction',
                                    ha="center", va="center")

            # then hide axis labels except left and bottom charts
            if i < num_columns - 1: ax[i][j].xaxis.set_visible(False)
            if j > 0: ax[i][j].yaxis.set_visible(False)

    # fix the bottom right and top left axis labels, which are wrong because
    # their charts only have text in them
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())

    plt.show()

def plot_bucket_histogram(points, bucket_size, title=None):
    histogram = dst.bucketize_points(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    y_max = 1.1 * max(histogram.values())
    if title:
        plt.title(title)
    plt.ylim(ymin=0, ymax=y_max)
    plt.show()
    
def corr_heatmap(df, out_name):
    corr = df.corr()

    # graph correlation matrix
    corr_fig, ax = plt.subplots() 
    corr_hm = sns.heatmap(corr, ax=ax,
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values)
    
    corr_hm.figure.savefig(out_name, bbox_inches='tight')
    
    return corr_hm