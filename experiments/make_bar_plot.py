
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

# Import figure making stuff
from matplotlib.font_manager import FontProperties
import seaborn as sns
color_names = ["black",
               "windows blue",
               "red",
               "amber",
               "medium green",
               "dusty purple",
               "orange"]

colors = sns.xkcd_palette(color_names)
sns.set_style("ticks")
sns.set_context("paper")


SIGMAS = [0.05, 0.25, 0.5, 0.75]
METHODS = ["True",
           "S.B.",
           "Rounding",
           "Mallows (0.5)",
           "Mallows (1.0)",
           "Mallows (5.0)"]

N_BARS_TO_PLOT = 6

if __name__ == "__main__":

    # TODO Load in the data here!!!
    # each key is a value of sigma
    # each value is a matrix of number of methods x number of permutations
    data = {
        0.05 : np.random.dirichlet(3 * np.ones(7), size=len(METHODS)),
        0.25 : np.random.dirichlet(3 * np.ones(7), size=len(METHODS)),
        0.50 : np.random.dirichlet(3 * np.ones(7), size=len(METHODS)),
        0.75 : np.random.dirichlet(3 * np.ones(7), size=len(METHODS)),
    }

    fig = plt.figure(figsize=(3.25, 3.25))

    fp = FontProperties()
    fp.set_weight("bold")

    barw = 1. / (len(METHODS) + 2)

    for i, sigma in enumerate(SIGMAS):
        data_i = data[sigma]

        ax = plt.subplot(len(SIGMAS), 1, i+1)
        for j, method in enumerate(METHODS):
            label = method if i == 0 else None
            ax.bar(np.arange(N_BARS_TO_PLOT) + j*barw,
                   data_i[j,:N_BARS_TO_PLOT],
                   width=barw,
                   color=colors[j],
                   edgecolor='k',
                   lw=.5,
                   label=label)

            # Plot the sum of the rest
            ax.bar(N_BARS_TO_PLOT + j * barw,
                   data_i[j, N_BARS_TO_PLOT:].sum(),
                   width=barw,
                   color=colors[j],
                   edgecolor='k',
                   lw=.5)

            # Plot dividing lines
            for n in range(N_BARS_TO_PLOT):
                xn = n + barw * (len(METHODS) + .5)
                ax.plot([xn, xn], [0, 1], ':k', lw=.5, alpha=0.5)

        # Add legend to top row
        if i == 0:
            plt.legend(loc="upper right", ncol=2, fontsize=6,
                       labelspacing=.5,
                       handlelength=1.5,
                       columnspacing=1.5)

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # ax.set_title("$\sigma = {}$".format(sigma)
        plt.text(0.5, .9, "$\sigma = {}$".format(sigma),
                 horizontalalignment='center',
                 fontsize=8,
                 transform=ax.transAxes)

        ax.set_ylabel("$\Pr(X)$")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1.0])

        # Set up xticks
        ax.set_xticks(np.arange(N_BARS_TO_PLOT + 1) + (len(METHODS) * barw) / 2.0)
        if i == len(SIGMAS) - 1:
            ax.set_xticklabels(list(map(str, np.arange(N_BARS_TO_PLOT)+1))
                               + ['>{}'.format(N_BARS_TO_PLOT)] )
            ax.set_xlabel("Permutation $X$ (sorted index)")
        else:
            ax.set_xticklabels([])

        plt.tight_layout(pad=0.1)

    plt.savefig("figure_2b.pdf")
    plt.savefig("figure_2b.png", dpi=300)
    plt.show()
