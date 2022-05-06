import tightbinding.moire_tb as mtb
import matplotlib.pyplot as plt


def tightbinding_plot_fulltb(n_moire:int, n_g:int, n_k:int, band:int, datatype:str, pathname:str):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, '0', disp=True, fulltb=True)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k-1], kline[3*n_k-1]])
    ax.set_xticklabels([r'$\bar{K}$',  r'$\bar{\Gamma}$',  r'$\bar{M}$', r'$\bar{K}^\prime$'])
    ax.set_xlim(0, kline[-1])

    for i in range(band):
        plt.plot(kline, emesh[:, n_band//2+i],'-', c='blue', lw=1)
        plt.plot(kline, emesh[:, n_band//2-1-i],'-', c='blue', lw=1)


    ax.set_ylabel("Engergy (eV)")
    ax.set_title("Full TB "+datatype+" Nmoire "+str(n_moire))
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k-1], color="black")
    ax.axvline(x=kline[2*n_k-1], color="black")
    ax.axvline(x=kline[3*n_k-1], color="black")

    plt.savefig(pathname+str(n_moire)+datatype+"_fulltb.png", dpi=500)



def tightbinding_plot_valley_comb(n_moire:int, n_g:int, n_k:int, band:int, datatype:str, pathname:str):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, 'valley_comb', disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k-1], kline[3*n_k-1]])
    ax.set_xticklabels([r'$\bar{K}$',  r'$\bar{\Gamma}$',  r'$\bar{M}$', r'$\bar{K}^\prime$'])
    ax.set_xlim(0, kline[-1])

    for i in range(band):
        plt.plot(kline, emesh[:, n_band//2+i],'-', c='blue', lw=1)
        plt.plot(kline, emesh[:, n_band//2-1-i],'-', c='blue', lw=1)


    ax.set_ylabel("Engergy (eV)")
    ax.set_title("TB Planewave: Combine Valley "+datatype+" Nmoire "+str(n_moire))
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k-1], color="black")
    ax.axvline(x=kline[2*n_k-1], color="black")
    ax.axvline(x=kline[3*n_k-1], color="black")

    plt.savefig(pathname+str(n_moire)+datatype+"_vcomb.png", dpi=500)



def tightbinding_plot_sep_valley(n_moire:int, n_g:int, n_k:int, band:int, datatype:str, pathname:str):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, '+1', disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k-1], kline[3*n_k-1]])
    ax.set_xticklabels([r'$\bar{K}$',  r'$\bar{\Gamma}$',  r'$\bar{M}$', r'$\bar{K}^\prime$'])
    ax.set_xlim(0, kline[-1])

    for i in range(band):
        plt.plot(kline, emesh[:, n_band//2+i],'-', c='blue', lw=1)
        plt.plot(kline, emesh[:, n_band//2-1-i],'-', c='blue', lw=1)

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, '-1', disp=True, fulltb=False)

    for i in range(band):
        plt.plot(kline, emesh[:, n_band//2+i],'-', c='blue', lw=1)
        plt.plot(kline, emesh[:, n_band//2-1-i],'-', c='blue', lw=1)

    ax.set_ylabel("Engergy (eV)")
    ax.set_title("TB Planewave: Separate Valley "+datatype+" Nmoire "+str(n_moire))
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k-1], color="black")
    ax.axvline(x=kline[2*n_k-1], color="black")
    ax.axvline(x=kline[3*n_k-1], color="black")

    plt.savefig(pathname+str(n_moire)+datatype+"_vsep.png", dpi=500)


def tb_sep_valley_cmp(n_moire:int, n_g:int, n_k:int, bandfull:int, bandpw: int, datatype:str, pathname:str):
    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, '0', disp=True, fulltb=True)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k-1], kline[3*n_k-1]])
    ax.set_xticklabels([r'$\bar{K}$',  r'$\bar{\Gamma}$',  r'$\bar{M}$', r'$\bar{K}^\prime$'])
    ax.set_xlim(0, kline[-1])

    for i in range(bandfull):
        plt.plot(kline, emesh[:, n_band//2+i],'-', c='red', lw=2, alpha=0.8)
        plt.plot(kline, emesh[:, n_band//2-1-i],'-', c='red', lw=2,alpha=0.8)

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, '+1', disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    for i in range(bandpw):
        plt.plot(kline, emesh[:, n_band//2+i],'-', c='blue', lw=1)
        plt.plot(kline, emesh[:, n_band//2-1-i],'-', c='blue', lw=1)

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, '-1', disp=True, fulltb=False)

    for i in range(bandpw):
        plt.plot(kline, emesh[:, n_band//2+i],'-', c='blue', lw=1)
        plt.plot(kline, emesh[:, n_band//2-1-i],'-', c='blue', lw=1)
    
    ax.set_ylabel("Engergy (eV)")
    ax.set_title("TB Planewave: Separate Valley VS FullTB "+datatype+" Nmoire "+str(n_moire))
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k-1], color="black")
    ax.axvline(x=kline[2*n_k-1], color="black")
    ax.axvline(x=kline[3*n_k-1], color="black")

    plt.savefig(pathname+str(n_moire)+datatype+"_vsep_cmp.png", dpi=500)


def tb_comb_valley_cmp(n_moire:int, n_g:int, n_k:int, bandfull:int, bandpw: int, datatype:str, pathname:str):

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, '0', disp=True, fulltb=True)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k-1], kline[3*n_k-1]])
    ax.set_xticklabels([r'$\bar{K}$',  r'$\bar{\Gamma}$',  r'$\bar{M}$', r'$\bar{K}^\prime$'])
    ax.set_xlim(0, kline[-1])

    for i in range(bandfull):
        plt.plot(kline, emesh[:, n_band//2+i],'-', c='red', lw=2, alpha=0.8)
        plt.plot(kline, emesh[:, n_band//2-1-i],'-', c='red', lw=2,alpha=0.8)

    emesh, _, kline, _, _ = mtb.tightbinding_solver(n_moire, n_g, n_k, datatype, 'valley_comb', disp=True, fulltb=False)
    n_band = emesh[0].shape[0]

    for i in range(bandpw):
        plt.plot(kline, emesh[:, n_band//2+i],'-', c='blue', lw=1)
        plt.plot(kline, emesh[:, n_band//2-1-i],'-', c='blue', lw=1)
    
    ax.set_ylabel("Engergy (eV)")
    ax.set_title("TB Planewave: Combine Valley VS FullTB "+datatype+" Nmoire "+str(n_moire))
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k-1], color="black")
    ax.axvline(x=kline[2*n_k-1], color="black")
    ax.axvline(x=kline[3*n_k-1], color="black")

    plt.savefig(pathname+str(n_moire)+datatype+"_vcomb_cmp.png", dpi=500)