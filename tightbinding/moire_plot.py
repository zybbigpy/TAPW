import tightbinding.moire_tb as mtb
import matplotlib.pyplot as plt
import numpy as np


def chemical_potential(emesh):
    n_band = emesh[0].shape[0]
    condunction_min = np.min(emesh[:, n_band // 2])
    vallence_max = np.max(emesh[:, n_band // 2-1])
    mu = (condunction_min+vallence_max)/2

    return mu


def tightbinding_plot_sparsetb(
        n_moire: int,
        n_g: int,
        n_k: int,
        band: int,
        datatype: str,
        pathname: str,
        name="",
        mu=False,
):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire,
                                                    n_g,
                                                    n_k,
                                                    datatype,
                                                    "0",
                                                    disp=True,
                                                    fulltb=True,
                                                    sparse=True)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])
    ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
    ax.set_xlim(0, kline[-1])

    if band>5:
        band = 5
    for i in range(band):
        plt.plot(kline, emesh[:, n_band // 2+i], ".", c="blue")
        plt.plot(kline, emesh[:, n_band // 2-1-i], ".", c="blue")

    ax.set_ylabel("Engergy (eV)")
    if mu == True:
        mu_val = chemical_potential(emesh)
        ax.axhline(y=mu_val, linewidth=1.5, linestyle="--", color="grey")
    ax.set_title(name)
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k], color="black")
    ax.axvline(x=kline[2*n_k], color="black")
    ax.axvline(x=kline[3*n_k], color="black")
    plt.tight_layout()
    plt.savefig(pathname+str(n_moire)+datatype+"_sparsetb.png", dpi=500)


def tightbinding_plot_fulltb(
        n_moire: int,
        n_g: int,
        n_k: int,
        band: int,
        datatype: str,
        pathname: str,
        name="",
        mu=False,
):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire,
                                                    n_g,
                                                    n_k,
                                                    datatype,
                                                    "0",
                                                    disp=True,
                                                    fulltb=True,
                                                    sparse=False)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])
    ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
    ax.set_xlim(0, kline[-1])

    for i in range(band):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    ax.set_ylabel("Engergy (eV)")
    if mu == True:
        mu_val = chemical_potential(emesh)
        ax.axhline(y=mu_val, linewidth=1.5, linestyle="--", color="grey")
    ax.set_title(name)
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k], color="black")
    ax.axvline(x=kline[2*n_k], color="black")
    ax.axvline(x=kline[3*n_k], color="black")
    plt.tight_layout()
    plt.savefig(pathname+str(n_moire)+datatype+"_fulltb.png", dpi=500)


def tightbinding_plot_valley_comb(
        n_moire: int,
        n_g: int,
        n_k: int,
        band: int,
        datatype: str,
        pathname: str,
        name="",
        mu=False,
):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, "valley_comb", disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])
    ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
    ax.set_xlim(0, kline[-1])

    for i in range(band):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    ax.set_ylabel("Engergy (eV)")
    if mu == True:
        mu_val = chemical_potential(emesh)
        ax.axhline(y=mu_val, linewidth=1.5, linestyle="--", color="grey")
    ax.set_title(name)
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k], color="black")
    ax.axvline(x=kline[2*n_k], color="black")
    ax.axvline(x=kline[3*n_k], color="black")
    plt.tight_layout()
    plt.savefig(pathname+str(n_moire)+datatype+"_vcomb.png", dpi=500)


def tightbinding_plot_sep_valley(
        n_moire: int,
        n_g: int,
        n_k: int,
        band: int,
        datatype: str,
        pathname: str,
        name="",
        mu=False,
):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, "+1", disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])
    ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
    ax.set_xlim(0, kline[-1])

    for i in range(band):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, "-1", disp=True, fulltb=False)

    for i in range(band):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    ax.set_ylabel("Engergy (eV)")
    if mu == True:
        mu_val = chemical_potential(emesh)
        ax.axhline(y=mu_val, linewidth=1.5, linestyle="--", color="grey")
    ax.set_title(name)
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k], color="black")
    ax.axvline(x=kline[2*n_k], color="black")
    ax.axvline(x=kline[3*n_k], color="black")
    plt.tight_layout()
    plt.savefig(pathname+str(n_moire)+datatype+"_vsep.png", dpi=500)


def tb_sep_valley_cmp(
        n_moire: int,
        n_g: int,
        n_k: int,
        bandfull: int,
        bandpw: int,
        datatype: str,
        pathname: str,
        name="",
):
    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire,
                                                    n_g,
                                                    n_k,
                                                    datatype,
                                                    "0",
                                                    disp=True,
                                                    fulltb=True,
                                                    sparse=False)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])
    ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
    ax.set_xlim(0, kline[-1])

    for i in range(bandfull):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="red", lw=2, alpha=0.8)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="red", lw=2, alpha=0.8)

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, "+1", disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    for i in range(bandpw):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, "-1", disp=True, fulltb=False)

    for i in range(bandpw):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    ax.set_ylabel("Engergy (eV)")
    ax.set_title(name)
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k], color="black")
    ax.axvline(x=kline[2*n_k], color="black")
    ax.axvline(x=kline[3*n_k], color="black")
    plt.tight_layout()
    plt.savefig(pathname+str(n_moire)+datatype+"_vsep_cmp.png", dpi=500)


def tb_comb_valley_cmp(
        n_moire: int,
        n_g: int,
        n_k: int,
        bandfull: int,
        bandpw: int,
        datatype: str,
        pathname: str,
        name="",
):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire,
                                                    n_g,
                                                    n_k,
                                                    datatype,
                                                    "0",
                                                    disp=True,
                                                    fulltb=True,
                                                    sparse=False)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])

    ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
    ax.set_xlim(0, kline[-1])

    for i in range(bandfull):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="red", lw=2, alpha=0.8)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="red", lw=2, alpha=0.8)

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, "valley_comb", disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    for i in range(bandpw):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    ax.set_ylabel("Engergy (eV)")
    ax.set_title(name)
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k], color="black")
    ax.axvline(x=kline[2*n_k], color="black")
    ax.axvline(x=kline[3*n_k], color="black")
    plt.tight_layout()
    plt.savefig(pathname+str(n_moire)+datatype+"_vcomb_cmp.png", dpi=500)


def sparsetb_sep_valley_cmp(
        n_moire: int,
        n_g: int,
        n_k: int,
        bandsparse: int,
        bandpw: int,
        datatype: str,
        pathname: str,
        name="",
):
    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, "+1", disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])
    ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
    ax.set_xlim(0, kline[-1])

    for i in range(bandpw):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, "-1", disp=True, fulltb=False)

    for i in range(bandpw):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    # In Sparse solver, we only reserve 15 kpts on the kline
    n_k = 5
    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire,
                                                    n_g,
                                                    n_k,
                                                    datatype,
                                                    "0",
                                                    disp=True,
                                                    fulltb=True,
                                                    sparse=True)
    n_band = emesh[0].shape[0]

    if bandsparse>5:
        bandsparse = 5

    for i in range(bandsparse):
        plt.plot(kline, emesh[:, n_band // 2+i], ".", c="red", alpha=0.8)
        plt.plot(kline, emesh[:, n_band // 2-1-i], ".", c="red", alpha=0.8)

    ax.set_ylabel("Engergy (eV)")
    ax.set_title(name)
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k], color="black")
    ax.axvline(x=kline[2*n_k], color="black")
    ax.axvline(x=kline[3*n_k], color="black")
    plt.tight_layout()
    plt.savefig(pathname+str(n_moire)+datatype+"_vsep_cmp_sparse.png", dpi=500)


def sparsetb_comb_valley_cmp(
        n_moire: int,
        n_g: int,
        n_k: int,
        bandsparse: int,
        bandpw: int,
        datatype: str,
        pathname: str,
        name="",
):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, "valley_comb", disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])
    ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
    ax.set_xlim(0, kline[-1])

    for i in range(bandpw):
        plt.plot(kline, emesh[:, n_band // 2+i], "-", c="blue", lw=1)
        plt.plot(kline, emesh[:, n_band // 2-1-i], "-", c="blue", lw=1)

    # we only compute 15 kpts on the kline for comparison
    n_k = 5
    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire,
                                                    n_g,
                                                    n_k,
                                                    datatype,
                                                    "0",
                                                    disp=True,
                                                    fulltb=True,
                                                    sparse=True)
    n_band = emesh[0].shape[0]

    if bandsparse>5:
        bandsparse = 5
    for i in range(bandsparse):
        plt.plot(kline, emesh[:, n_band // 2+i], ".", c="red", alpha=0.8)
        plt.plot(kline, emesh[:, n_band // 2-1-i], ".", c="red", alpha=0.8)

    ax.set_ylabel("Engergy (eV)")
    ax.set_title(name)
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k], color="black")
    ax.axvline(x=kline[2*n_k], color="black")
    ax.axvline(x=kline[3*n_k], color="black")
    plt.tight_layout()
    plt.savefig(pathname+str(n_moire)+datatype+"_vcomb_cmp_sparse.png", dpi=500)
