import mtbmtbg.moire_tb as mtb
import matplotlib.pyplot as plt
import numpy as np

from mtbmtbg.config import DataType, EngineType, ValleyType


def chemical_potential(emesh: np.ndarray) -> float:
    """determine the feimi energy for an insulator

    Args:
        emesh (np.ndarray): eigenvalues return by tb solver

    Returns:
        float: fermi energy
    """

    n_band = emesh[0].shape[0]
    condunction_min = np.min(emesh[:, n_band//2])
    vallence_max = np.max(emesh[:, n_band//2-1])
    mu = (condunction_min+vallence_max)/2

    return mu


def plot_module(ax: plt.axes,
                kline: np.ndarray,
                emesh: np.ndarray,
                n_k: int,
                bands: int,
                shape: str = "-",
                color: str = "blue",
                alpha: float = 1,
                figname: str = "",
                mu: bool = False):
    """plot module for TBG

    Args:
        ax (plt.axes): ax object
        kline (np.ndarray): kline
        emesh (np.ndarray): band eigen value
        n_k (int): number of kpnts on each path
        bands (int): number of bands plotted above or below fermi energy
        shape (str, optional): shape. Defaults to "-".
        color (str, optional): color. Defaults to "blue".
    """

    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])
    ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
    ax.set_xlim(0, kline[-1])
    ax.set_ylabel("Engergy (eV)")
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k], color="black")
    ax.axvline(x=kline[2*n_k], color="black")
    ax.axvline(x=kline[3*n_k], color="black")

    n_band = emesh[0].shape[0]
    for i in range(bands):
        ax.plot(kline, emesh[:, n_band//2+i], shape, c=color, lw=1, alpha=alpha)
        ax.plot(kline, emesh[:, n_band//2-1-i], shape, c=color, lw=1, alpha=alpha)

    if mu == True:
        mu_val = chemical_potential(emesh)
        ax.axhline(y=mu_val, linewidth=1.5, linestyle="--", color="grey")

    ax.set_title(figname)


def tb_plot_sparsetb(n_moire: int, n_g: int, n_k: int, bands: int, datatype: str, pathname="./", figname="", mu=False):
    fig, ax = plt.subplots()
    ret = mtb.tb_solver(n_moire, n_g, n_k, True, datatype, engine=EngineType.TBSPARSE)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k, bands, shape='.', figname=figname, mu=mu)
    plt.tight_layout()
    plt.savefig(pathname+"moire_"+str(n_moire)+"_"+datatype+"_sparsetb.png", dpi=500)


def tb_plot_fulltb(n_moire: int, n_g: int, n_k: int, bands: int, datatype: str, pathname="./", figname="", mu=False):
    fig, ax = plt.subplots()
    ret = mtb.tb_solver(n_moire, n_g, n_k, True, datatype, engine=EngineType.TBFULL)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k, bands, figname=figname, mu=mu)
    plt.tight_layout()
    plt.savefig(pathname+"moire_"+str(n_moire)+"_"+datatype+"_fulltb.png", dpi=500)


def tb_plot_tbplw_sepv(n_moire: int, n_g: int, n_k: int, bands: int, datatype: str, pathname="./", figname="",
                       mu=False):
    fig, ax = plt.subplots()
    ret = mtb.tb_solver(n_moire, n_g, n_k, True, datatype, valley=ValleyType.VALLEY1)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k, bands, figname=figname, mu=mu)
    ret = mtb.tb_solver(n_moire, n_g, n_k, True, datatype, valley=ValleyType.VALLEY2)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k, bands, figname=figname, mu=mu)
    plt.tight_layout()
    plt.savefig(pathname+"moire_"+str(n_moire)+"_"+datatype+"_.tbplw_sepv.png", dpi=500)


def tb_plot_tbplw_combv(n_moire: int,
                        n_g: int,
                        n_k: int,
                        bands: int,
                        datatype: str,
                        pathname="./",
                        figname="",
                        mu=False):
    fig, ax = plt.subplots()
    ret = mtb.tb_solver(n_moire, n_g, n_k, True, datatype, valley=ValleyType.VALLEYC)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k, bands, figname=figname, mu=mu)
    plt.tight_layout()
    plt.savefig(pathname+"moire_"+str(n_moire)+"_"+datatype+"_.tbplw_combv.png", dpi=500)


def fulltb_sepv_cmp(n_moire: int,
                    n_g: int,
                    n_k1: int,
                    n_k2: int,
                    band1: int,
                    band2: int,
                    datatype: str,
                    pathname="./",
                    figname=""):
    fig, ax = plt.subplots()
    ret = mtb.tb_solver(n_moire, n_g, n_k1, True, datatype, valley=ValleyType.VALLEY1)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k1, band1, figname=figname)
    ret = mtb.tb_solver(n_moire, n_g, n_k1, True, datatype, valley=ValleyType.VALLEY2)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k1, band1, figname=figname)
    ret = mtb.tb_solver(n_moire, n_g, n_k2, True, datatype, engine=EngineType.TBFULL)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k2, band2, shape='.', color='red', alpha=0.5, figname=figname)
    plt.tight_layout()
    plt.savefig(pathname+"moire_"+str(n_moire)+"_"+datatype+"_.fulltb_sepv_cmp.png", dpi=500)


def fulltb_combv_cmp(n_moire: int,
                     n_g: int,
                     n_k1: int,
                     n_k2: int,
                     band1: int,
                     band2: int,
                     datatype: str,
                     pathname="./",
                     figname=""):
    fig, ax = plt.subplots()
    ret = mtb.tb_solver(n_moire, n_g, n_k1, True, datatype, valley=ValleyType.VALLEYC)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k1, band1, figname=figname)
    ret = mtb.tb_solver(n_moire, n_g, n_k2, True, datatype, engine=EngineType.TBFULL)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k2, band2, shape='.', color='red', alpha=0.5, figname=figname)
    plt.tight_layout()
    plt.savefig(pathname+"moire_"+str(n_moire)+"_"+datatype+"_.fulltb_combv_cmp.png", dpi=500)


def sparsetb_sepv_cmp(n_moire: int,
                      n_g: int,
                      n_k1: int,
                      n_k2: int,
                      band1: int,
                      band2: int,
                      datatype: str,
                      pathname="./",
                      figname=""):
    fig, ax = plt.subplots()
    ret = mtb.tb_solver(n_moire, n_g, n_k1, True, datatype, valley=ValleyType.VALLEY1)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k1, band1, figname=figname)
    ret = mtb.tb_solver(n_moire, n_g, n_k1, True, datatype, valley=ValleyType.VALLEY2)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k1, band1, figname=figname)
    ret = mtb.tb_solver(n_moire, n_g, n_k2, True, datatype, engine=EngineType.TBSPARSE)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k2, band2, shape='.', color='red', alpha=0.5, figname=figname)
    plt.tight_layout()
    plt.savefig(pathname+"moire_"+str(n_moire)+"_"+datatype+"_.sparsetb_sepv_cmp.png", dpi=500)


def sparsetb_combv_cmp(n_moire: int,
                       n_g: int,
                       n_k1: int,
                       n_k2: int,
                       band1: int,
                       band2: int,
                       datatype: str,
                       pathname="./",
                       figname=""):
    fig, ax = plt.subplots()
    ret = mtb.tb_solver(n_moire, n_g, n_k1, True, datatype, valley=ValleyType.VALLEYC)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k1, band1, figname=figname)
    ret = mtb.tb_solver(n_moire, n_g, n_k2, True, datatype, engine=EngineType.TBSPARSE)
    emesh = ret['emesh']
    kline = ret['kline']
    plot_module(ax, kline, emesh, n_k2, band2, shape='.', color='red', alpha=0.5, figname=figname)
    plt.tight_layout()
    plt.savefig(pathname+"moire_"+str(n_moire)+"_"+datatype+"_.sparsetb_combv_cmp.png", dpi=500)
