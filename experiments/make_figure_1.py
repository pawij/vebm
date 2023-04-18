import itertools as it

import numpy as np
np.random.seed(0)
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import figure making stuff
from matplotlib.font_manager import FontProperties
import seaborn as sns
color_names = ["windows blue",
               "red",
               "amber",
               "medium green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "mint",
               "salmon",
               "dark brown"]

colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("paper")

from hips.plotting.layout import create_axis_at_location


import birkhoff.simplex as simplex
from birkhoff.primitives import psi_to_pi, logistic
from birkhoff.utils import get_b3_projection, project_perm_to_sphere
from birkhoff.primitives import psi_to_birkhoff


WIDTH, HEIGHT = 6.6, 3.3
K = 3
N_marks = 6
N_smpls = 1000
SMPL_KWARGS = dict(ls='', marker='o', markersize=4, color=colors[0], alpha=0.1)
MARK_KWARGS = dict(marker='o', markersize=5, markeredgecolor='k', mew=0.5)

def make_figure_1():
    # Set up figure dims
    pad = 0.15
    w_ax = WIDTH / 4.0
    w_pan = w_ax - 2 * pad
    h_header = 0.3
    h_half = (HEIGHT - h_header) / 2.0
    h_pan = h_half - 2 * pad

    print("w_pan: ", w_pan, " h_pan: ", h_pan)

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    fp = FontProperties()
    fp.set_weight("bold")

    ax_a = create_axis_at_location(fig, 0 * w_ax + pad, h_half + pad, w_pan, h_pan, projection="3d")
    ax_b = create_axis_at_location(fig, 0 * w_ax + pad, pad, w_pan, h_pan)
    plot_softmax_psi_pi(ax_a, ax_b)

    plt.figtext(pad / WIDTH,
                (HEIGHT - .1) / HEIGHT,
                "(a)",
                horizontalalignment="center",
                verticalalignment="top",
                fontproperties=fp)

    plt.figtext((pad + 0.5 * w_pan) / WIDTH,
                (HEIGHT - .1) / HEIGHT,
                "Gumbel-softmax",
                horizontalalignment="center",
                verticalalignment="top",
                fontproperties=fp)

    # Make c a little smaller
    shrink = 0.8
    ax_c = create_axis_at_location(fig,
                                   1 * w_ax + pad + (1-shrink)/2.*w_pan,
                                   h_half + pad + (1-shrink)/2.*h_pan,
                                   shrink * w_pan, shrink * h_pan)
    ax_d = create_axis_at_location(fig, 1 * w_ax + pad, pad, w_pan, h_pan)
    plot_discrete_stickbreak_psi_pi(ax_c, ax_d)

    plt.figtext((pad + w_ax) / WIDTH,
                (HEIGHT - .1) / HEIGHT,
                "(b)",
                horizontalalignment="center",
                verticalalignment="top",
                fontproperties=fp)

    plt.figtext((1 * w_ax + pad + 0.5 * w_pan) / WIDTH,
                (HEIGHT - .1) / HEIGHT,
                "Stick-breaking\n(categorical)",
                horizontalalignment="center",
                verticalalignment="top",
                fontproperties=fp)

    ax_e = create_axis_at_location(fig, 2 * w_ax + pad, h_half + pad, w_pan, h_pan, projection="3d")
    ax_f = create_axis_at_location(fig, 2 * w_ax + pad, pad, w_pan, h_pan)
    plot_stickbreak_psi_pi(fig, ax_e, ax_f)

    plt.figtext((pad + 2 * w_ax) / WIDTH,
                (HEIGHT - .1) / HEIGHT,
                "(c)",
                horizontalalignment="center",
                verticalalignment="top",
                fontproperties=fp)

    plt.figtext((2 * w_ax + pad + 0.5 * w_pan) / WIDTH,
                (HEIGHT - .1) / HEIGHT,
                "Stick-breaking\n(permutations)",
                horizontalalignment="center",
                verticalalignment="top",
                fontproperties=fp)

    ax_g = create_axis_at_location(fig, 3 * w_ax + pad, h_half + pad, w_pan, h_pan)
    ax_h = create_axis_at_location(fig, 3 * w_ax + pad, pad, w_pan, h_pan)
    plot_round_psi_pi(fig, ax_g, ax_h)

    plt.figtext((pad + 3 * w_ax) / WIDTH,
                (HEIGHT - .1) / HEIGHT,
                "(d)",
                horizontalalignment="center",
                verticalalignment="top",
                fontproperties=fp)

    plt.figtext((3 * w_ax + pad + 0.5 * w_pan) / WIDTH,
                (HEIGHT - .1) / HEIGHT,
                "Rounding\n(permutations)",
                horizontalalignment="center",
                verticalalignment="top",
                fontproperties=fp)

    plt.savefig("figure1.pdf")
    plt.savefig("figure1.png", dpi=300)
    plt.show()

def _find_mark_psis_with_kmeans(psi_smpls):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N_marks)
    km.fit(psi_smpls)
    return km.cluster_centers_

###
### Discrete softmax
###
def _sample_softmax_psi_pi():
    softmax = lambda x: np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    psi_smpls = np.random.gumbel(0, 1, size=(N_smpls, 3))
    pi_smpls = softmax(psi_smpls)
    xy_smpls = simplex.proj_to_2D(pi_smpls)

    psi_marks = _find_mark_psis_with_kmeans(psi_smpls)
    pi_marks = softmax(psi_marks)
    xy_marks = simplex.proj_to_2D(pi_marks)
    return psi_marks, pi_marks, xy_marks, psi_smpls, pi_smpls, xy_smpls


def plot_softmax_psi_pi(ax_psi, ax_pi):
    psi_marks, pi_marks, xy_marks, psi_smpls, pi_smpls, xy_smpls = _sample_softmax_psi_pi()

    lim = (-1, 5)
    ax_psi.plot(psi_smpls[:, 0], psi_smpls[:, 1], psi_smpls[:, 2], **SMPL_KWARGS)

    # Plot labels on top of cloud
    ax_psi.plot(lim, [0, 0], [0, 0], '-k')
    ax_psi.plot([0, 0], lim, [0, 0], '-k')
    ax_psi.plot([0, 0], [0, 0], lim, '-k')

    for i, psi in enumerate(psi_marks):
        ax_psi.plot([psi[0]], [psi[1]], [psi[2]], color=colors[i + 1], **MARK_KWARGS)

    ax_psi.set_xlim(lim)
    ax_psi.set_ylim(lim)
    ax_psi.set_zlim(lim)
    ax_psi.set_xticklabels([])
    ax_psi.set_yticklabels([])
    ax_psi.text(lim[1] + .25, 0, 0, '$\\psi_1$', fontsize=8)
    ax_psi.text(0, lim[1] + .25, 0, '$\\psi_2$', fontsize=8)
    ax_psi.text(0, 0, lim[1] + .25, '$\\psi_3$', fontsize=8)
    ax_psi.axis('off')
    ax_psi.set_aspect("equal")

    ### Plot simplex samples
    ax_pi.plot(xy_smpls[:, 0], xy_smpls[:, 1], 'o', **SMPL_KWARGS)
    simplex.plot_simplex(ax_pi)

    for i, xy in enumerate(xy_marks):
        ax_pi.plot(xy[0], xy[1], color=colors[i+1], **MARK_KWARGS)
    ax_pi.set_aspect("equal")

###
### Discrete stick-breaking
###
def _sample_stickbreak_psi_pi():
    # psi_smpls = np.random.rand(N_smpls, 2)
    psi_smpls = logistic(2 * np.random.randn(N_smpls, 2))
    pi_smpls = np.array([psi_to_pi(psi) for psi in psi_smpls])
    xy_smpls = np.array([simplex.proj_to_2D(pi) for pi in pi_smpls])

    psi_marks = _find_mark_psis_with_kmeans(psi_smpls)
    pi_marks = np.array([psi_to_pi(psi) for psi in psi_marks])
    xy_marks = simplex.proj_to_2D(pi_marks)

    return psi_marks, pi_marks, xy_marks, psi_smpls, pi_smpls, xy_smpls

def plot_discrete_stickbreak_psi_pi(ax_psi, ax_pi):
    psi_marks, pi_marks, xy_marks, psi_smpls, pi_smpls, xy_smpls = _sample_stickbreak_psi_pi()

    r = [0, 1]
    for s, e in it.combinations(np.array(list(it.product(r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax_psi.plot(*zip(s, e), '-k')
    ax_psi.text(-0.1, .05, "0", fontsize=6)
    ax_psi.text(-0.1, .95, "1", fontsize=6)
    ax_psi.text(-0.2, 0.5, "$\\beta_{2}$", fontsize=8)

    ax_psi.text(.05, -0.1, "0", fontsize=6)
    ax_psi.text(.95, -0.1, "1", fontsize=6)
    ax_psi.text(0.5, -0.2, "$\\beta_{1}$", fontsize=8)
    ax_psi.axis('off')
    ax_psi.set_aspect("equal")

    ax_psi.plot(psi_smpls[:, 0], psi_smpls[:, 1], **SMPL_KWARGS)
    for i, psi in enumerate(psi_marks):
        ax_psi.plot([psi[0]], [psi[1]], color=colors[i+1], **MARK_KWARGS)

    ### Plot simplex
    ax_pi.plot(xy_smpls[:, 0], xy_smpls[:, 1], 'o', **SMPL_KWARGS)
    simplex.plot_simplex(ax_pi)
    for i, xy in enumerate(xy_marks):
        ax_pi.plot(xy[0], xy[1], color=colors[i + 1], **MARK_KWARGS)
    ax_pi.set_aspect("equal")

###
### Permutation stickbreaking
###
def _plot_birkhoff_projection(fig, ax, Q):
    # For K = 3 there are only 6 permutations. Plot each of their
    # projections.
    Ps = []
    for perm in it.permutations(np.arange(K)):
        P = np.zeros((K, K))
        P[np.arange(K), np.array(perm)] = 1
        Ps.append(P)
    Ps = np.array(Ps)
    Psis = np.array([project_perm_to_sphere(P, Q) for P in Ps])

    # Get the convex hull of the points
    from scipy.spatial import ConvexHull
    hull = ConvexHull(Psis[:, :2])
    for simplex in hull.simplices:
        for i in range(2):
            st = Psis[simplex[i]]
            en = Psis[simplex[(i + 1) % 2]]
            ax.plot([st[0], en[0]], [st[1], en[1]], '-k', lw=1)


    # Plot little stencils of the permutation matrices
    ax_pos = ax.get_position()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    nx = lambda x: (x - xlim[0]) / (xlim[1] - xlim[0])
    ny = lambda y: (y - ylim[0]) / (ylim[1] - ylim[0])
    subw = 0.075

    for i, perm in enumerate(it.permutations(np.arange(K))):
        x, y = Psis[i, 0], Psis[i, 1]
        subax = create_axis_at_location(fig,
                                        (ax_pos.x0 + nx(x) * ax_pos.width) * WIDTH - subw,
                                        (ax_pos.y0 + ny(y) * ax_pos.height) * HEIGHT - subw,
                                        2 * subw, 2 * subw,
                                        ticks=False)
        subax.patch.set_color('none')
        Pi = np.zeros((K, K))
        Pi[np.arange(K), perm] = 1
        subax.imshow(Pi, interpolation="nearest")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def _sample_stickbreak_perm_psi_pi(Q):
    # Sample a bunch of points in the Birkhoff polytope
    # psi_smpls = np.random.rand(N_smpls, K - 1, K - 1)
    psi_smpls = logistic(2 * np.random.randn(N_smpls, K - 1, K - 1))
    # psi_smpls[:, 1, 1] = 0.
    pi_smpls = []
    xy_smpls = []
    for n, psi in enumerate(psi_smpls):
        pi = psi_to_birkhoff(psi)
        xy = project_perm_to_sphere(pi, Q)
        pi_smpls.append(pi)
        xy_smpls.append(xy)
    pi_smpls = np.array(pi_smpls)
    xy_smpls = np.array(xy_smpls)

    # Get some marks
    psi_marks = _find_mark_psis_with_kmeans(psi_smpls.reshape((N_smpls, -1))).reshape((N_marks, K-1, K-1))
    # pi_marks = np.array([psi_to_pi(psi) for psi in psi_marks])
    # xy_marks = simplex.proj_to_2D(pi_marks)
    pi_marks = []
    xy_marks = []
    for n, psi in enumerate(psi_marks):
        pi = psi_to_birkhoff(psi)
        xy = project_perm_to_sphere(pi, Q)
        pi_marks.append(pi)
        xy_marks.append(xy)
    pi_marks = np.array(pi_marks)
    xy_marks = np.array(xy_marks)

    return psi_marks, pi_marks, xy_marks, psi_smpls, pi_smpls, xy_smpls


def plot_stickbreak_psi_pi(fig, ax_psi, ax_pi, axes=((0, 0), (0, 1), (1, 0))):
    # Now plot the trajectory in the K-1 x K-1 unit hypercube

    rs = np.random.get_state()
    np.random.seed(1)
    Q = get_b3_projection()
    np.random.set_state(rs)

    psi_marks, pi_marks, xy_marks, psi_smpls, pi_smpls, xy_smpls = _sample_stickbreak_perm_psi_pi(Q)

    # ax.plot(fUs[:,axes[0]], fUs[:, axes[1]], fUs[:, axes[2]], ':k')
    ax_psi.plot(psi_smpls[:, axes[0][0], axes[0][1]],
                psi_smpls[:, axes[1][0], axes[1][1]],
                psi_smpls[:, axes[2][0], axes[2][1]],
                **SMPL_KWARGS)

    for i, psi in enumerate(psi_marks):
        ax_psi.plot([psi[axes[0]]], [psi[axes[1]]], [psi[axes[2]]],
                    color=colors[i+1], **MARK_KWARGS)

    ax_psi.text(-0.1, -0.1, -0.1,  "0", fontsize=6)
    # ax_psi.text(.95, -0.1, "1", fontsize=6)
    ax_psi.text(0.5, -0.4, 0., "$\\beta_{11}$", fontsize=8)
    ax_psi.text(1.2, 0.5, 0., "$\\beta_{12}$", fontsize=8)
    ax_psi.text(-0.3, -0.2, 0.5, "$\\beta_{21}$", fontsize=8)
    ax_psi.axis('off')
    ax_psi.set_aspect("equal")

    # Draw the cube
    r = [0, 1]
    for s, e in it.combinations(np.array(list(it.product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax_psi.plot(*zip(s, e), '-k')

    # Draw the simplex
    _plot_birkhoff_projection(fig, ax_pi, Q)
    ax_pi.plot(xy_smpls[:, 0], xy_smpls[:, 1], **SMPL_KWARGS)
    for i, xy in enumerate(xy_marks):
        ax_pi.plot(xy[0], xy[1], color=colors[i+1], **MARK_KWARGS)

    ax_pi.axis('off')
    ax_pi.set_aspect("equal")


def _sample_round_psi_pi(Q, eta, tau):
    psi_smpls = 1. / K * np.ones((K, K)) + eta * np.random.randn(N_smpls, K, K)
    pi_smpls, xy_smpls = [], []
    for n, Psi in enumerate(psi_smpls):
        xy_smpls.append(project_perm_to_sphere(Psi, Q))

        # Round to the nearest permutation
        rows, cols = linear_sum_assignment(-Psi)
        B_nearest = np.zeros((K, K))
        B_nearest[rows, cols] = 1
        B = tau * Psi + (1 - tau) * B_nearest
        pi_smpls.append(project_perm_to_sphere(B, Q))
    xy_smpls = np.array(xy_smpls)
    pi_smpls = np.array(pi_smpls)

    # psi_marks = _find_mark_psis_with_kmeans(psi_smpls.reshape((N_smpls, -1))).reshape((N_marks, K, K))
    Ps = []
    for perm in it.permutations(np.arange(K)):
        P = np.zeros((K, K))
        P[np.arange(K), np.array(perm)] = 1
        Ps.append(P)
    Ps = np.array(Ps)
    psi_marks = .75 * np.ones((K, K)) / K + (1 - .75) * Ps

    pi_marks, xy_marks = [], []
    for n, Psi in enumerate(psi_marks):
        xy_marks.append(project_perm_to_sphere(Psi, Q))

        # Round to the nearest permutation
        rows, cols = linear_sum_assignment(-Psi)
        B_nearest = np.zeros((K, K))
        B_nearest[rows, cols] = 1
        B = tau * Psi + (1 - tau) * B_nearest
        pi_marks.append(project_perm_to_sphere(B, Q))
    xy_marks = np.array(xy_marks)
    pi_marks = np.array(pi_marks)
    
    return psi_marks, pi_marks, xy_marks, psi_smpls, pi_smpls, xy_smpls


def plot_round_psi_pi(fig, ax_psi, ax_pi, tau=0.5, eta=0.4):
    rs = np.random.get_state()
    np.random.seed(1)
    Q = get_b3_projection()
    np.random.set_state(rs)

    psi_marks, pi_marks, xy_marks, psi_smpls, pi_smpls, xy_smpls = _sample_round_psi_pi(Q, eta, tau)

    # Draw the simplex
    _plot_birkhoff_projection(fig, ax_psi, Q)
    ax_psi.plot(xy_smpls[:, 0], xy_smpls[:, 1], **SMPL_KWARGS)
    for i, xy in enumerate(xy_marks):
        ax_psi.plot(xy[0], xy[1], color=colors[i+1], **MARK_KWARGS)
    ax_psi.axis('off')
    ax_psi.set_aspect("equal")

    _plot_birkhoff_projection(fig, ax_pi, Q)
    ax_pi.plot(pi_smpls[:, 0], pi_smpls[:, 1], **SMPL_KWARGS)
    for i, pi in enumerate(pi_marks):
        ax_pi.plot(pi[0], pi[1], color=colors[i + 1], **MARK_KWARGS)
    ax_pi.axis('off')
    ax_pi.set_aspect("equal")

    #
    # # inds_to_plot = np.arange(0, N, step=N//10)
    # # # ax.plot(xys[:,0], xys[:,1], ':k')
    # ax_psi.plot(xy_smpls[:, 0], xy_smpls[:, 1], **SMPL_KWARGS)
    # for i, xy in enumerate(xy_marks):
    #     ax_psi.plot(xy[0], xy[1],
    #             'o', markersize=8, color=colors[i])
    #
    # ax_psi.axis('off')
    # # fig.savefig('sb_birkhoff.pdf')
    # # fig.savefig('sb_birkhoff.png')
    #
    # ############################
    # fig = create_figure(figsize=(2.7, 2.5), transparent=True)
    # ax = create_axis_at_location(fig, 0.1, 0.1, 2.5, 2.3, transparent=True)
    #
    # # Get the convex hull of the points
    # from scipy.spatial import ConvexHull
    # hull = ConvexHull(vertices[:, :2])
    # for simplex in hull.simplices:
    #     for i in range(2):
    #         st = vertices[simplex[i]]
    #         en = vertices[simplex[(i + 1) % 2]]
    #         ax.plot([st[0], en[0]], [st[1], en[1]], '-k', lw=1)
    #
    # # Plot little stencils of the permutation matrices
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # subw = 0.1
    # nx = lambda x: 2.5 * (x - xlim[0]) / (xlim[1] - xlim[0])
    # ny = lambda y: 2.3 * (y - ylim[0]) / (ylim[1] - ylim[0])
    # for i, perm in enumerate(it.permutations(np.arange(K))):
    #     x, y = vertices[i, 0], vertices[i, 1]
    #     subax = create_axis_at_location(fig, 0.1 + nx(x) - subw,
    #                                     0.1 + ny(y) - subw,
    #                                     2 * subw, 2 * subw,
    #                                     ticks=False)
    #     subax.patch.set_color('none')
    #     Pi = np.zeros((K, K))
    #     Pi[np.arange(K), perm] = 1
    #     subax.imshow(Pi, interpolation="nearest")
    #
    # # inds_to_plot = np.arange(0, N, step=N//10)
    # # # ax.plot(xys[:,0], xys[:,1], ':k')
    # ax.plot(xy_samples[:, 0], xy_samples[:, 1], 'ko', markersize=4, alpha=0.1)
    # for i, xy in enumerate(xy_marker_trans):
    #     ax.plot(xy[0], xy[1],
    #             'o', markersize=8, color=colors[i])
    #
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    # ax.axis('off')
    # # fig.savefig('sb_birkhoff.pdf')
    # fig.savefig('sb_birkhoff.png')


if __name__ == "__main__":
    make_figure_1()