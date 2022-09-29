import matplotlib.pyplot as plt
import matplotlib as mplt
from matplotlib.widgets import Slider
from funcs import *

mplt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
mplt.rc('text', usetex=True)
mplt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

rest_lambda = ang_freq_to_lambda(1)
obsSpeed = 0.001
obsPolar = (0.7,4)
observerVector = sph2cart(obsPolar)
observerVector2 = np.asarray([observerVector])
n_points = 10**3

xi, yi, zi = sample_spherical(n_points)
initialPointsVectors = np.asarray([xi, yi, zi]).T

rotatedPoints, shift, obsLambda = transformedPoints(
    obsSpeed, observerVector2, rest_lambda, initialPointsVectors,DopplerShift=True)

fig, ax = plt.subplots(
    1, 1, subplot_kw={'projection': '3d', 'aspect': 'auto'})
ax.set_box_aspect((np.ptp(xi), np.ptp(yi), np.ptp(zi)))
scat = ax.scatter(
    rotatedPoints[0:len(rotatedPoints), 0],
    rotatedPoints[0:len(rotatedPoints), 1],
    rotatedPoints[0:len(rotatedPoints), 2],
    s=10,
    # c = RGBs/255.0,
    c=shift,
    cmap='rainbow',
    vmin=-1,
    vmax=1,
    zorder=10)

cbar = plt.colorbar(scat, location='left')
cbar.set_label('Doppler shift $\Delta \lambda / \lambda$')

# Remove tick labels from axes
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# plt.subplots_adjust(bottom=0)

# Make sliders to control the speed, azimuthal angle and polar angle of the observer's motion.
axSpeed = plt.axes([0.4, 0.11, 0.45, 0.03])
speed_slider = Slider(
    ax=axSpeed,
    label='Speed ($c$)',
    valmin=0,
    valmax=0.9999999,
    valinit=obsSpeed,
)

axAzimuthal = plt.axes([0.4, 0.06, 0.45, 0.03])
azimuthal_slider = Slider(
    ax=axAzimuthal,
    label=r'$\phi$ (rad)',
    valmin=0,
    valmax=(np.pi)*2,
    valinit=obsPolar[1],
)

axPolar = plt.axes([0.4, 0.01, 0.45, 0.03])
polar_slider = Slider(
    ax=axPolar,
    label=r'$\theta$ (rad)',
    valmin=0,
    valmax=np.pi,
    valinit=obsPolar[0],
)

# set to metropolis background colour
fig.set_facecolor('#fafafa')
ax.set_facecolor('#fafafa')
ax.view_init(elev=22)

# The function to be called anytime a slider's value changes
def update(val):
    obsPolar = (polar_slider.val,azimuthal_slider.val)
    rotatedPoints, shift, obsLambda = transformedPoints(speed_slider.val, np.asarray(
        [sph2cart(obsPolar)]), rest_lambda, initialPointsVectors,DopplerShift=True)  # changed obsPolar so it plays nice
    scat._offsets3d = (
        rotatedPoints[0:len(rotatedPoints),0],
        rotatedPoints[0:len(rotatedPoints),1],
        rotatedPoints[0:len(rotatedPoints),2])
    # np.c_[RGBs, 255*np.ones(len(RGBs))] # add alpha channel—RGBA.
    # scat._facecolor3d = RGBs/255.0
    # scat._edgecolor3d = RGBs/255.0
    # scat.set_array(RGBs/255.0)
    # scat.set_color(RGBs/255.0)
    scat.set_array(shift)
    fig.canvas.draw()

speed_slider.on_changed(update)
azimuthal_slider.on_changed(update)
polar_slider.on_changed(update)

# for loop for gif
speed_range = np.linspace(0.001,0.999,120)
iteration = 0
for v in speed_range:
    iteration += 1
    speed_slider.set_val(v)
    obsPolar = (polar_slider.val,azimuthal_slider.val)
    rotatedPoints, shift, obsLambda = transformedPoints(speed_slider.val, np.asarray(
        [sph2cart(obsPolar)]), rest_lambda, initialPointsVectors,DopplerShift=True)  # changed obsPolar so it plays nice
    scat._offsets3d = (
        rotatedPoints[0:len(rotatedPoints),0],
        rotatedPoints[0:len(rotatedPoints),1],
        rotatedPoints[0:len(rotatedPoints),2])
    scat.set_array(shift)

    plt.savefig('Report Figures/points-on-sphere-gif/frame' + str(iteration) + '.png',dpi=200)
    
plt.show()