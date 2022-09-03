import healpy as hp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import numpy as np

### LaTeX Font for plotting ###
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
##########

nside = 16
npix = hp.nside2npix(nside)
healpy_map = np.zeros(npix)

tests_pol = [0.7, np.pi/2, np.pi-0.7, np.pi/2 - 30*np.pi/180]
tests_az = [4,4,4-np.pi/2,np.pi/2]
labels = ['']

lon1 = (-np.pi,np.pi)
lat1 = (30*np.pi/180,30*np.pi/180)

lon2 = (-np.pi,np.pi)
lat2 = (-30*np.pi/180,-30*np.pi/180)

x = [x - np.pi for x in tests_az]
y = [np.pi/2 - y for y in tests_pol]

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
            color='tab:blue',
            label='Masked region boundary')

plt.plot(
            lon2,
            lat2,
            linestyle='dashed',
            color='tab:blue')

plt.fill_between(
            lon1,
            lat1[0],
            lat2[0],
            alpha=0.2,
            label='Masked region')
plt.scatter(
            x,
            y,
            marker='x',
            color='tab:orange',
            label='Tested point')

plt.scatter(
            2*np.pi - 264.021 * np.pi/180,
            48.253*np.pi/180,
            marker='x',
            color='tab:red',
            label='CMB dipole (Planck Collaboration 2020)')

plt.title(
            'Points on the Sky Tested',
            fontsize=14)

plt.legend(
            loc='lower right',
            fontsize=12)

plt.savefig('locations.pdf')
