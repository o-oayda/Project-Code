import healpy as hp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import numpy as np
from funcs import conv2GalacticCoords

### LaTeX Font for plotting ###
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
##########

nside = 16
npix = hp.nside2npix(nside)
healpy_map = np.zeros(npix)

az = [4, 4, 4 - np.pi, np.pi/2]
pol = [0.7, np.pi/2, np.pi - 0.7, np.pi/2 - 30*np.pi/180]
# convert coordinate system (see function documentation)
az, pol = conv2GalacticCoords(az,pol)
labels = ['']

lon1 = (-np.pi,np.pi)
lat1 = (30*np.pi/180,30*np.pi/180)

lon2 = (-np.pi,np.pi)
lat2 = (-30*np.pi/180,-30*np.pi/180)

pol_CMB, az_CMB = [48.253*np.pi/180], [264.021 * \
    np.pi/180]

# the polar value is already properly adjusted so polar_conv is set to false
# (0.73,4.6)
az_CMB = conv2GalacticCoords(az_CMB,pol_CMB,polar_conv=False)

# old
# x = [x - np.pi for x in tests_az]
# y = [np.pi/2 - y for y in tests_pol]

hp.projview(healpy_map,
            projection_type='mollweide',
            coord=["G"],
            graticule=True,
            graticule_labels=True,
            color="white",
            cbar=False,
            longitude_grid_spacing=30)

# masked region
plt.plot(
            lon1,
            lat1,
            linestyle='dashed',
            color='#23373b',
            label='Masked region boundary',
            alpha=0.5)

plt.plot(
            lon2,
            lat2,
            linestyle='dashed',
            color='#23373b',
            alpha=0.7)

plt.fill_between(
            lon1,
            lat1[0],
            lat2[0],
            # color='#23373b',
            alpha=0.15,
            label='Masked region')

# plt.scatter takes in (az, pol)
plt.scatter(
            az,
            pol,
            marker='x',
            color='tab:orange',
            label='Tested point',
            zorder=10)

plt.scatter(
            az_CMB,
            pol_CMB,
            marker='x',
            color='tab:red',
            label='CMB dipole (Planck Collaboration 2020)',
            zorder=10)

plt.title(
            'Points on the Sky Tested',
            fontsize=17)

plt.legend(
            loc='lower right',
            fontsize=12)

fig = plt.gcf()
ax = plt.gca()

# fig.set_facecolor('#fafafa')
# ax.set_facecolor('#fafafa')

labs = ['A','B','C','D']

for i, txt in enumerate(labs):
    ax.annotate(txt,
    (az[i]+0.05, pol[i]-0.2),
    bbox=dict(
        facecolor='lightgrey',
        edgecolor='black',
        boxstyle='round,pad=0.5',
        alpha=0.8),
    fontsize=13)

plt.savefig('Report Figures/locations-tested-report.pdf',bbox_inches='tight')
