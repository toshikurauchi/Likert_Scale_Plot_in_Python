'''
Created on March 20, 2017

@author: Diako Mardanbegi <d.mardanbegi@lancaster.ac.uk>
'''

import seaborn as sns
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
from pylab import *
import io
import os
import sys
import types
from collections import defaultdict
pd.options.mode.chained_assignment = None  # default='warn'
sns.set_style("whitegrid")
mpl.rc("savefig", dpi=150)


# Constants
FONT1 = {
    'family': 'sans-serif',
    'color': 'white',
    'weight': 'normal',
    'size': 10,
}
FONT2 = {
    'family': 'sans-serif',
    'color': 'grey',
    'weight': 'normal',
    'size': 8,
}
EXTRA_SHIFT = 0.5
BARWIDTH = 0.2
WHITE = (1.0, 1.0, 1.0)
SPACES_AFTER_LABEL = 4
RIGHT_MARGIN = 2


def check_valid(retval=None):
    def wrap(func):
        def wrapper(self, *args, **kwargs):
            if not self.valid:
                return retval
            return func(self, *args, **kwargs)
        return wrapper
    return wrap


class LikertData(object):
    def __init__(self, tb, n_point, ax, **kwargs):
        self.tb = tb
        self.ax = ax
        self.n_point = n_point
        self.condition_column = kwargs.get('condition_column', 'condition')
        self.reverse_colors = kwargs.get('reverse_colors', False)
        self._invalidate()

    def _invalidate(self):
        self._condition_dfs = None
        self._conditions = None
        self._questions = None
        self._total_participants = None
        self._colors = None
        self._middles_all = None

    @property
    @check_valid(0)
    def total_participants(self):
        if self._total_participants is None:
            self._total_participants = len(self.tb)
            if self.condition_column is not None:
                self._total_participants /= len(self.conditions)
        return self._total_participants

    @property
    @check_valid(list())
    def questions(self):
        if self._questions is None:
            self._questions = self.df.columns.tolist()
            if self.condition_column in self._questions:
                self._questions.remove(self.condition_column)
            else:
                self._questions = self._questions[::-1]
        return self._questions

    @property
    @check_valid(list())
    def conditions(self):
        if self._conditions is None:
            if self.condition_column is None:
                self._conditions = []
            else:
                self._conditions = self.df[self.condition_column].unique()
        return self._conditions

    @property
    @check_valid(list())
    def condition_dfs(self):
        if self._condition_dfs is None:
            self._condition_dfs = []

            for cond in self.conditions:
                # Count values for each question
                cond_rows = self.df[self.condition_column] == cond
                cdf = self.df.loc[cond_rows, self.questions]
                self._condition_dfs.append(self._condition_df(cdf))
            if self.condition_column is None:
                self._condition_dfs = [
                    self._condition_df(self.df.loc[:, self.questions])[::-1]
                ]

        return self._condition_dfs

    def _condition_df(self, df):
        cond_counts = df.apply(pd.Series.value_counts)
        return cond_counts.reindex(self.range).T.fillna(0)

    @property
    @check_valid(list())
    def middles_all(self):
        if self._middles_all is None:
            self._middles_all = [
                c[self.half_range].sum(axis=1) + c[self.n_point // 2 + 1] * .5
                for c in self.condition_dfs
            ]
        return self._middles_all

    @property
    def tb(self):
        return self._tb

    @tb.setter
    def tb(self, new_tb):
        self._tb = new_tb
        if not self.valid:
            return
        self.df = self._tb.copy(deep=True)
        self._invalidate()

    @property
    def colors(self):
        if self._colors is None:
            self._colors = sns.color_palette("coolwarm", self.n_point)
            if self.reverse_colors:
                self._colors = list(reversed(self._colors))
            self._colors = [WHITE] + self._colors
        return self._colors

    @property
    def range(self):
        return list(range(1, self.n_point + 1))

    @property
    def half_range(self):
        return list(range(1, self.n_point // 2 + 1))

    def middle(self, condition):
        return self.middles_all[condition]

    @property
    @check_valid(0)
    def longest_middle(self):
        return np.array(self.middles_all).max()

    @property
    @check_valid(0)
    def complete_longest(self):
        return max([(df_c.sum(axis=1) - self.middle(cond)).max()
                    for cond, df_c in enumerate(self.condition_dfs)])

    def comp_y(self, y, cond, dashed=False):
        n = len(self.conditions)
        if n > 0:
            return y + (float(cond) * 1.0 - float(n) + 1) * BARWIDTH
        elif dashed:
            return -1.3 - y * (2 * BARWIDTH + EXTRA_SHIFT) + BARWIDTH / 2
        return -1.3 - (y + 0.1) * (2 * BARWIDTH + EXTRA_SHIFT)

    @property
    def valid(self):
        return self._tb is not None

    def plot_condition(self, cond, longest, patches_already_moved, total):
        df = self.condition_dfs[cond]
        df.insert(0, '', (self.middle(cond) - longest).abs())

        patch_handles = df.plot.barh(ax=self.ax, stacked=True, color=self.colors,
                                     legend=False, width=BARWIDTH,
                                     edgecolor='white')

        patches = [p for p in patch_handles.get_children()
                   if _good_patch(p, patches_already_moved)]

        for p in patches:
            p.set_xy((p.get_x(), self.comp_y(p.get_y(), cond)))

            if p.get_width() <= 1 or p.get_facecolor()[0: 3] == WHITE:
                continue

            percent = p.get_width() / self.total_participants * 100
            patch_handles.text(
                p.get_x() + p.get_width() / 2,
                p.get_y() + p.get_height() / total,
                "{0:.0f}%".format(percent),
                ha="center",
                fontdict=FONT1
            )

        return patches_already_moved + patches

    def plot_dashed(self, cond, min_x):
        for i in range(0, len(self.questions)):
            y = self.comp_y(i, cond, True)
            if cond < len(self.conditions):
                self.ax.text(
                    min_x - 0.7,
                    y - BARWIDTH / 4.0,
                    self.conditions[cond],
                    ha="center",
                    fontdict=FONT2)
            self.ax.plot([min_x, self.condition_dfs[cond].iloc[i, 0]], [y, y],
                         linestyle=':', color='grey', alpha=.2, linewidth=1)


def plot_likert_over_conditions(tb, n_point, custom_likert_range=None, tb2=None,
                                custom_likert_range2=None, ax=None, **kwargs):
    '''
    This function takes a table of questions and their responses in likert scale
    (1:positive N:negative) as columns, as well as a another column indicating
    the condition of the response.

    This function also takes another table for general questions after all
    conditions.

    Other keyword arguments:
    reverse_colors: (bool) if True, reverses colors in the scale
    condition_column: (str) name of the condition column (default: 'condition')
    '''

    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ld = LikertData(tb, n_point, ax, **kwargs)
    kwargs2 = kwargs.copy()
    kwargs2['condition_column'] = None
    gd = LikertData(tb2, n_point, ax, **kwargs2)
    custom_likert_labels_by_y_axis = []

    # --------------------------------------------------------------------
    # add shift column to each table
    longest = max(ld.longest_middle, gd.longest_middle)
    complete_longest = int(longest + max(ld.complete_longest,
                                         gd.complete_longest))

    patches_already_moved = []
    for cond, df_c in enumerate(ld.condition_dfs):
        patches_already_moved = ld.plot_condition(cond, longest,
                                                  patches_already_moved,
                                                  len(ld.questions))

    yticks = list(ax.get_yticks())

    custom_likert_labels_by_y_axis = _range2label(custom_likert_range,
                                                  len(ld.questions))

    if gd.valid:
        g_lab = _range2label(custom_likert_range2, len(gd.questions))[:: -1]
        custom_likert_labels_by_y_axis = g_lab + custom_likert_labels_by_y_axis

        # Plotting general questions
        g_yticks = [gd.comp_y(i, 0, True)
                    for i in range(len(gd.questions) - 1, -1, -1)]
        yticks = g_yticks + yticks

        patches_already_moved = gd.plot_condition(0, longest,
                                                  patches_already_moved,
                                                  len(ld.questions))
        gd.plot_dashed(0, -5)

    z = ax.axvline(longest, linestyle='-', color='black', alpha=.5,
                   linewidth=1)
    z.set_zorder(-1)

    plt.xlim(-5, complete_longest + RIGHT_MARGIN)
    ymin = -1 * len(gd.condition_dfs[0] if gd.condition_dfs else []) - 1
    plt.ylim(ymin, len(ld.questions) - 0.5)

    xvalues = range(0, complete_longest, 10)
    plt.xticks(xvalues, [])
    plt.xlabel('Percentage', fontsize=12, horizontalalignment='left')
    xlabel_x = float(longest) / (complete_longest + RIGHT_MARGIN)
    ax.xaxis.set_label_coords(xlabel_x, -0.01)

    ylabels = gd.questions + ld.questions

    plt.yticks(yticks, [yl + ' ' * SPACES_AFTER_LABEL for yl in ylabels])

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    # adding condition indicators on the y axis
    for cond, df_c in enumerate(ld.condition_dfs):
        ld.plot_dashed(cond, ax.get_xlim()[0] + 0.5 + 0.7)

    plt.grid('off')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # adding Likert range legend
    draw_legend(yticks, custom_likert_labels_by_y_axis, ax, ld.colors)

    plt.show()


def _g(x, y):
    return x.loc[y] if y in x.index else 0


def _good_patch(p, patches_already_moved):
    return isinstance(p, (matplotlib.patches.Rectangle)) and \
        p not in patches_already_moved and p.get_height() == BARWIDTH


def _range2label(custom_likert_range, n_questions):
    if custom_likert_range is None:
        custom_likert_range = dict()
    custom_likert_range = defaultdict(lambda: ['very low', 'very high'],
                                      custom_likert_range)
    return [custom_likert_range[key] for key in range(1, n_questions + 1)]


def _make_bbox(facecolor):
    return {
        'edgecolor': 'none',
        'facecolor': facecolor,
        'alpha': 1.0,
        'pad': 2
    }


def draw_legend(yticks, custom_likert_labels_by_y_axis, ax, colors):
    for i, y_tick in enumerate(yticks):
        v = custom_likert_labels_by_y_axis[i]
        x = -12

        y = yticks[i] - 0.4
        ax.text(x, y, v[0], fontsize=8, zorder=6, color='white',
                horizontalalignment='right', bbox=_make_bbox(colors[1]))

        for ci, c in enumerate(colors[1:-1]):
            x = x + 0.3
            ax.text(x, y, '-', fontsize=8, zorder=6, color=c,
                    horizontalalignment='right', bbox=_make_bbox(c))

        ax.text(x + 0.2, y, v[1], fontsize=8, zorder=6, color='white',
                horizontalalignment='left', bbox=_make_bbox(colors[-1]))
