import dynesty.plotting as dyplot
import matplotlib.pyplot as plt
import healpy as hp
import dynesty.plotting as dyplot
import numpy as np
from matplotlib import rc
import matplotlib
from astropy.visualization import hist
from matplotlib.pyplot import xlabel

def uncertaintyOnSkyPlot(nside,dresults,az_true,pol_true,n):
    npix = hp.nside2npix(nside)
    healpy_map = np.zeros(npix)

    ### LaTeX Font for plotting ###
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
    rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    ##########

    hp.projview(healpy_map,
                projection_type='mollweide',
                coord=["G"],
                graticule=True,
                graticule_labels=True,
                color="white",
                cbar=False,
                longitude_grid_spacing=30)

    # the try statement is due to an annoying set xlim TypeError error which recommended me to use cartopy
    try:
        # ax = plt.figure()
        # ax.add_subplot(projection='mollweide')
        plt.grid(True)
        dyplot._hist2d(
            dresults.samples[:, 1] - np.pi,  # x: phi
            np.pi/2 - dresults.samples[:, 2],  # y: theta
            weights=np.exp(dresults['logwt']-dresults['logz']
                        [-1])  # line 1246 plotting.py
        )
    except TypeError:
        plt.scatter(
            dresults.samples[-1, 1] - np.pi, np.pi/2 - dresults.samples[-1, 2],
            marker='x',
            color='orange',
            label='Maximum likelihood fitted point')
        plt.scatter(
            2*np.pi - 264.021 * np.pi/180,
            48.253*np.pi/180,
            marker='x',
            color='tab:blue',
            label='CMB dipole (Planck Collaboration 2020)')
        plt.scatter(
            az_true - np.pi,
            np.pi/2 - pol_true,
            marker='x',
            color='tab:red',
            label='True direction of motion'
        )
        plt.legend(loc='lower right')
        plt.title('Uncertainty in Location of Dipole (Galactic Coordinates)')
        pass

    plt.savefig('Results/n' + str(n) + '-uncertainty-on-sky.pdf')


def numberCountDensity(hpx_map,m,n):
    m[m == 0] = hp.UNSEEN
    hpx_map[np.where(m == hp.UNSEEN)] = 0
    ticks = [np.mean(hpx_map[hpx_map != 0])]

    image = hp.projview(hp.ma(hpx_map, badval=0),
                        title='Number Count Density Map of Simulated Sample',
                        projection_type='mollweide',
                        graticule=True,
                        graticule_labels=True,
                        longitude_grid_spacing=30,
                        cbar=False
                        # unit='Points per pixel',
                        )

    fig = plt.gcf()
    ax = plt.gca()
    middle_val = (min(hpx_map[hpx_map != 0]) + max(hpx_map))/2

    cbar = fig.colorbar(image,
                        location='left',
                        ax=ax,
                        shrink=0.7,
                        anchor=(1.0, 0.5),
                        label='Points per pixel',
                        ticks=[
                            min(hpx_map[hpx_map != 0]),
                            max(hpx_map),
                            middle_val]
                        )

    cbar.ax.hlines(np.mean(hpx_map[hpx_map != 0]), 0, 1,
                colors='k',
                linewidth=1
                )
    cbar.set_label(label='Points per pixel', size=11)
    cbar.ax.text(1.3, np.mean(hpx_map[hpx_map != 0])-4, '${:.0f}$'.format(
        np.mean(hpx_map[hpx_map != 0])))

    plt.savefig('Results/n' + str(n) + '-number-count-density.pdf')


def numberCountHist(hpx_map,n):
    fig, ax = plt.subplots(1, 1)

    ax.hist = hist(hpx_map[hpx_map != 0],
                bins='scott',
                histtype='step',
                color='k')

    ax.set_xlabel('Points per pixel')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Points Per Pixel across the Sky')

    plt.savefig('Results/n' + str(n) + '-number-count-histogram.pdf')
