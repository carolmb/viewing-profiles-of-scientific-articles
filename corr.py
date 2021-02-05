import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
from read_file import select_original_breakpoints


def plot_scatter_with_freq(X, Y, title, xlabel, ylabel, filename):
    df = pd.DataFrame({'x': X, 'y': Y})
    ax = sns.jointplot(x="x", y="y", data=df, kind="kde")
    ax.set_axis_labels(xlabel, ylabel, fontsize=18)

    # plt.suptitle(title)
    print(title)

    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    f_input = 'segm/segmented_curves_html.txt'
    slopes, intervals = select_original_breakpoints(5, f_input)

    intervals1 = intervals[:, 0]
    slopes1 = slopes[:, 0]

    X = intervals1
    Y = slopes1

    corr = r"$\rho$ = %.2f" % np.corrcoef(intervals1, slopes1)[1][0]

    plot_scatter_with_freq(intervals1, slopes1, corr, r'$l_1$', r'$\alpha_1$', 'alpha_l_1')

    intervals1 = intervals[:, 1]
    slopes1 = slopes[:, 1]

    X = np.concatenate([X, intervals1], axis=0)
    Y = np.concatenate([Y, slopes1], axis=0)

    corr = r"$\rho$ = %.2f" % np.corrcoef(intervals1, slopes1)[1][0]

    plot_scatter_with_freq(intervals1, slopes1, corr, r'$l_2$', r'$\alpha_2$', 'alpha_l_2')

    intervals1 = intervals[:, 2]
    slopes1 = slopes[:, 2]

    X = np.concatenate([X, intervals1], axis=0)
    Y = np.concatenate([Y, slopes1], axis=0)

    corr = r"$\rho$ = %.2f" % np.corrcoef(intervals1, slopes1)[1][0]

    plot_scatter_with_freq(intervals1, slopes1, corr, r'$l_3$', r'$\alpha_3$', 'alpha_l_3')

    intervals1 = intervals[:, 3]
    slopes1 = slopes[:, 3]

    X = np.concatenate([X, intervals1], axis=0)
    Y = np.concatenate([Y, slopes1], axis=0)

    corr = r"$\rho$ = %.2f" % np.corrcoef(intervals1, slopes1)[1][0]

    plot_scatter_with_freq(intervals1, slopes1, corr, r'$l_4$', r'$\alpha_4$', 'alpha_l_4')

    intervals1 = intervals[:, 4]
    slopes1 = slopes[:, 4]

    X = np.concatenate([X, intervals1], axis=0)
    Y = np.concatenate([Y, slopes1], axis=0)

    corr = r"$\rho$ = %.2f" % np.corrcoef(intervals1, slopes1)[1][0]

    plot_scatter_with_freq(intervals1, slopes1, corr, r'$l_5$', r'$\alpha_5$', 'alpha_l_5')

    corr = r"$\rho$ = %.2f" % np.corrcoef(X, Y)[1][0]

    plot_scatter_with_freq(X, Y, corr, r'$l$', r'$\alpha$', 'alpha_l')

    plt.figure(figsize=(4, 3))
    sns.distplot(X)
    plt.suptitle(r'$l$')
    plt.savefig('l_dist.pdf')
    plt.clf()

    plt.figure(figsize=(4, 3))
    sns.distplot(Y)
    plt.suptitle(r'$\alpha$')
    plt.savefig('alpha_dist.pdf')
