import numpy as np
import scipy as scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

# plotting settings
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2,
      'axes.labelsize': 18,
      'axes.titlesize': 18,
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

# more parameters
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14


def noise(sigma=0.05, iters=1000):
    return np.random.normal(0, sigma, iters)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def coords_to_plot(x, y, entry, subset=None, **kwargs):
    if subset:
        index = np.random.randint(0, len(x[:, entry]), subset)
        r = x[:, entry][index]
        c = y[index]

        sorter = np.argsort(r)
        X = r[sorter]
        Y = c[sorter] + noise(iters=subset)
        return X, Y
    else:
        sorter = np.argsort(x[:, entry])
        X = x[:, entry][sorter]
        Y = y[sorter] + noise(**kwargs)
        return X, Y


def interpol(classes, classifications, entry, s=10):
    r, c = coords_to_plot(entry)
    f = scipy.interpolate.UnivariateSpline(running_mean(r, 10),
                                           running_mean(c, 10), s=s)
    xs = np.linspace(50, 1750, 50)
    return xs, f(xs)


def run_mean(classes, classifications, entry, interval, **kwargs):
    r, c = coords_to_plot(classes, classifications, entry, **kwargs)
    return running_mean(r, interval), running_mean(c, interval)


def plot_signal(r, l, c, ax, **kwargs):
    """"""
    def find(classification):
        x = r[np.where(c == classification)]
        y = l[np.where(c == classification)]
        return (x, y + np.random.normal(0, .1, len(y)))

    types = {b'true_0': 'ro',
             b'true_1': 'kv',
             b'false_0': 'bD',
             b'false_1': 'gp'}
    labels = {b'true_0': 'Empty',
              b'true_1': 'Full',
              b'false_0': 'Empty called Full',
              b'false_1': 'Full called empty'}
    for i, category in enumerate(np.unique(np.sort(c))):
        ax.plot(*find(category), types[category], label=labels[category],
                **kwargs)
    return ax


def make_pretty_plots(noisy_M, real_M, signals, calls,
                      iters=1000, **kwargs):
    """"""
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 10), sharex='col',
                           sharey='row')
    titles = ['Singly labelled class', 'Doubly labelled class',
              'Triply labelled class']
    cols = [0, 5, 6]

    for j, b in enumerate(ax):
        for i, a in enumerate(b):
            if j == 0:
                true0 = np.where(calls[:, cols[i]] == b'true_0')
                true1 = np.where(calls[:, cols[i]] == b'true_1')
                y = np.zeros(len(calls[:, cols[i]]))
                y[true0] = 1
                y[true1] = 1
                a.plot(*coords_to_plot(noisy_M, y, cols[i],
                       subset=500, iters=iters),
                       'o', ms=3, alpha=0.35)
                a.plot(*run_mean(noisy_M, y, cols[i], 50, iters=iters),
                       'r')
                a.set_title(titles[i])
            if j == 1:
                coords = (noisy_M[:, cols[i]],
                          real_M[:, cols[i]],
                          calls[:, cols[i]])
                a = plot_signal(*coords, ax=a, **kwargs)
            if j == 2:
                coords = (noisy_M[:, cols[i]],
                          signals[:, cols[i]],
                          calls[:, cols[i]])
                a = plot_signal(*coords, ax=a, **kwargs)

    ax[0, 0].set_ylabel('Classification')
    ax[1, 0].set_ylabel('Real class size')
    ax[2, 0].set_ylabel('Signal')
    ax[2, 1].set_xlabel('Noisy Class Size')

    if np.abs(signals).max() > 10*np.abs(signals).min():
        ax[2, 0].set_yscale('symlog')

    texts = ["Empty", "Full",
             'Empty called Full', 'Full called Empty']
    types = ['ro', 'kv', 'bD', 'gp']

    patches = [plt.plot([], [], types[i], ms=10, ls="",
               label="{:s}".format(texts[i]))[0] for i in range(len(types))]
    plt.legend(handles=patches,
               loc=(1, 1), ncol=1, numpoints=1)

    return fig, ax
