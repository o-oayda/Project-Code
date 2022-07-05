import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

num_steps = 10**5

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# empty arrays to contain x and y positions
x_trace = np.zeros(num_steps)
y_trace = np.zeros(num_steps)

# Initial location of star
x_trace[0] = 0.5
y_trace[0] = 0.5

# empty arrays for velocity
x_vel = np.zeros(num_steps)
y_vel = np.zeros(num_steps)

# Initial velocity of the star
x_vel[0] = 1
y_vel[0] = 1

# time step
dt = 10**(-3)


def force(x, y):
  force_x = -2*x/(1+x**2+y**2/2)
  force_y = -y/(1+x**2+y**2/2)
  return force_x, force_y


for i in range(1, len(x_trace)):
  # shift in position
  x_trace[i] = x_trace[i-1] + x_vel[i-1]*dt
  y_trace[i] = y_trace[i-1] + y_vel[i-1]*dt

  # change in velocity
  ## call force function
  force_x, force_y = force(x_trace[i-1], y_trace[i-1])

  ## update velocity
  x_vel[i] = x_vel[i-1] + force_x*dt
  y_vel[i] = y_vel[i-1] + force_y*dt

# plot final result
# fig, ax = plt.subplots()


# abs velocity
v = np.sqrt(x_vel**2 + y_vel**2)

# colour map
points = np.array([x_trace, y_trace]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
plt.plot(x_trace, y_trace, linewidth=0.25)
# Create a continuous norm to map from data points to colors
norm = plt.Normalize(v.min(), v.max())
lc = LineCollection(segments, cmap='jet', norm=norm)
# Set the values used for colormapping
lc.set_array(v)
lc.set_linewidth(3)
line = axs.add_collection(lc)
fig.colorbar(line, ax=axs)
# colour map

# angular momentum
L = v*(np.sqrt(x_trace**2 + y_trace**2))
print(L)

axs.set(xlabel='$x$ position', ylabel='$y$ position')
axs.set(xlim=(-2, 2), ylim=(-2, 2))
axs.set_aspect('equal')

plt.savefig('orbit-potentials.pdf',bbox_inches = 'tight')

plt.show()
