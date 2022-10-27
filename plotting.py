import dynesty.plotting as dyplot
import matplotlib.pyplot as plt
import healpy as hp
import dynesty.plotting as dyplot
import numpy as np
from matplotlib import rc
import matplotlib
from astropy.visualization import hist
from funcs import conv2GalacticCoords
from matplotlib.transforms import Bbox

def uncertaintyOnSkyPlot(nside,dresults,az_true,pol_true,n,resultsDir,topRightQuad=False):
    npix = hp.nside2npix(nside)
    healpy_map = np.zeros(npix)

    phi_hist = dresults.samples[:,1]
    theta_hist = dresults.samples[:,2]

    az_max = [dresults.samples[-1, 1]]
    pol_max = [dresults.samples[-1, 2]]

    ### LaTeX Font for plotting ###
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
    rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    ##########
    
    y_labs = []
    for i in range(1,6):
        val = -90 + 30 * i
        if i == 4:
            y_labs.append('$b=' + str(val) + r'^\circ$')
        else:
            y_labs.append('$' + str(val) + r'^\circ$')
    
    x_labs = []
    for i in range(1,12):
        if i < 6:
            val = -180 + 30 * i
        if i == 6:
            x_labs.append('$l=0' + r'^\circ$')
        else:
            val = 540 - 30 * i 
            x_labs.append('$' + str(val) + r'^\circ$')

    if topRightQuad:

        hp.projview(healpy_map,
                    projection_type='mollweide',
                    coord=["G"],
                    graticule=True,
                    graticule_labels=True,
                    color="white",
                    cbar=False,
                    longitude_grid_spacing=30,
                    custom_ytick_labels=y_labs,
                    custom_xtick_labels=x_labs)

        # the try statement is due to an annoying set xlim TypeError error which recommended me to use cartopy
        try:
            # ax = plt.figure()
            # ax.add_subplot(projection='mollweide')

            phi_hist, az_hist = conv2GalacticCoords(
                phi_hist,
                theta_hist,
                polar_conv=True)

            plt.grid(True)
            dyplot._hist2d(
                # old plotting
                # dresults.samples[:, 1] - np.pi,  # x: phi
                # np.pi/2 - dresults.samples[:, 2],  # y: theta
                phi_hist,
                az_hist,
                weights=np.exp(dresults['logwt']-dresults['logz']
                            [-1]))  # line 1246 plotting.py
        except TypeError:

            az_max, pol_max = conv2GalacticCoords(
                az_max,
                pol_max,
                polar_conv=True)
            
            pol_CMB, az_CMB = [48.253*np.pi/180], [264.021 * \
                np.pi/180]
            az_CMB = conv2GalacticCoords(
                az_CMB,
                pol_CMB,
                polar_conv=False)

            az_true, pol_true = conv2GalacticCoords(
                [az_true],
                [pol_true])

            plt.scatter(
                # dresults.samples[-1, 1] - np.pi,
                #  np.pi/2 - dresults.samples[-1, 2],
                az_max,
                pol_max,
                marker='x',
                color='orange',
                label='Max. likelihood point',
                zorder=10)

            plt.scatter(
                # 2*np.pi - 264.021 * np.pi/180,
                # 48.253*np.pi/180,
                az_CMB,
                pol_CMB,
                marker='x',
                color='tab:blue',
                label='CMB dipole',
                zorder=10)
        
            plt.scatter(
                # az_true - np.pi,
                # np.pi/2 - pol_true,
                az_true,
                pol_true,
                marker='x',
                color='tab:red',
                label='True direction',
                zorder=10)

            plt.legend(
                loc='upper left',
                ncol=3,
                bbox_to_anchor=(0.46,0.485))
            plt.title('Uncertainty in Location of Dipole (Gal. Coord.)',loc='right')

            fig = plt.gcf()
            ax = plt.gca()

            # bounding box for savefig
            x0,x1 = -0.25, np.pi + 0.45
            y0,y1 = -0.22, np.pi / 2
            bbox = Bbox([[x0,y0],[x1,y1]])
            bbox = bbox.transformed(ax.transData).transformed(fig.dpi_scale_trans.inverted())

            # position of tick labels
            ax.yaxis.tick_right()
            ax.tick_params(axis=u'both', which=u'both',length=0)
            
            # fig.set_facecolor('#fafafa')
            # ax.set_facecolor('#fafafa')

            plt.savefig('Results/' + resultsDir + '/n' + str(n) + '-uncertainty-on-sky.pdf',
                bbox_inches=bbox)
            pass
    
    else:
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

            phi_hist, az_hist = conv2GalacticCoords(
                phi_hist,
                theta_hist,
                polar_conv=True)

            plt.grid(True)
            dyplot._hist2d(
                # old plotting
                # dresults.samples[:, 1] - np.pi,  # x: phi
                # np.pi/2 - dresults.samples[:, 2],  # y: theta
                phi_hist,
                az_hist,
                weights=np.exp(dresults['logwt']-dresults['logz']
                                [-1]))  # line 1246 plotting.py
        except TypeError:

            az_max, pol_max = conv2GalacticCoords(
                az_max,
                pol_max,
                polar_conv=True)

            pol_CMB, az_CMB = [48.253*np.pi/180], [264.021 *
                                                    np.pi/180]
            az_CMB = conv2GalacticCoords(
                az_CMB,
                pol_CMB,
                polar_conv=False)

            az_true, pol_true = conv2GalacticCoords(
                [az_true],
                [pol_true])

            plt.scatter(
                # dresults.samples[-1, 1] - np.pi,
                #  np.pi/2 - dresults.samples[-1, 2],
                az_max,
                pol_max,
                marker='x',
                color='orange',
                label='Maximum likelihood fitted point',
                zorder=10)

            plt.scatter(
                # 2*np.pi - 264.021 * np.pi/180,
                # 48.253*np.pi/180,
                az_CMB,
                pol_CMB,
                marker='x',
                color='tab:blue',
                label='CMB dipole (Planck Collaboration 2020)',
                zorder=11)

            plt.scatter(
                # az_true - np.pi,
                # np.pi/2 - pol_true,
                az_true,
                pol_true,
                marker='x',
                color='tab:red',
                label='True direction of motion',
                zorder=12)

            plt.legend(loc='lower right')
            plt.title(
                'Uncertainty in Location of Dipole (Gal. Coord.)', loc='right')

            fig = plt.gcf()
            ax = plt.gca()
            ax.set_axisbelow(True)

            # fig.set_facecolor('#fafafa')
            # ax.set_facecolor('#fafafa')
            plt.savefig('Results/' + resultsDir + '/n' + str(n) +
                        '-uncertainty-on-sky.pdf')
            pass

def numberCountDensity(hpx_map,m,n,resultsDir):
    m[m == 0] = hp.UNSEEN
    hpx_map[np.where(m == hp.UNSEEN)] = 0
    ticks = [np.mean(hpx_map[hpx_map != 0])]

    image = hp.projview(hp.ma(hpx_map, badval=0),
                        title='Number Count Density Map of Simulated Sample',
                        projection_type='mollweide',
                        graticule=True,
                        graticule_labels=True,
                        longitude_grid_spacing=30,
                        cbar=False,
                        fontsize= {"title": 22, "xtick_label": 17, "ytick_label": 17})
                        # unit='Points per pixel'

    fig = plt.gcf()
    ax = plt.gca()
    middle_val = (min(hpx_map[hpx_map != 0]) + max(hpx_map))/2

    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.03])

    cbar = fig.colorbar(image,
                        orientation='horizontal',
                        # location='left',
                        ax=ax,
                        cax=cbar_ax,
                        shrink=0.7,
                        # anchor=(1.0, 0.5),
                        label='Points per pixel',
                        ticks=[
                            min(hpx_map[hpx_map != 0]),
                            max(hpx_map),
                            middle_val])

    cbar.ax.vlines(np.mean(hpx_map[hpx_map != 0]), 0, 1,
                colors='k',
                linewidth=1)

    cbar.set_label(label='Points per pixel', size=17)
    cbar.ax.text(
        np.mean(hpx_map[hpx_map != 0])-20,
        1.5,
        '$\\mu = {:.0f}$'.format(
            np.mean(hpx_map[hpx_map != 0])),
        fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # fig.set_facecolor('#fafafa')
    # ax.set_facecolor('#fafafa')
    # plt.title(
    'Number Count Density Map of Simulated Sample',
    # fontsize=16)

    plt.savefig('Results/' + resultsDir + '/n' + str(n) + '-number-count-density.pdf',
        bbox_inches='tight',
        dpi=300)


def numberCountHist(hpx_map,n,resultsDir):
    fig, ax = plt.subplots(1, 1)

    ax.hist = hist(hpx_map[hpx_map != 0],
                bins='scott',
                histtype='step',
                color='k')

    ax.set_xlabel(
        'Points per pixel',
        fontsize=16)

    ax.set_ylabel(
        'Frequency',
        fontsize=16)
        
    ax.set_title(
        'Histogram of Points Per Pixel across the Sky',
        fontsize=18)
    ax.tick_params(labelsize=15)
    
    # fig.set_facecolor('#fafafa')
    # ax.set_facecolor('#fafafa')

    plt.savefig('Results/' + resultsDir + '/n' + str(n) + '-number-count-histogram.pdf',
        bbox_inches='tight')
