# %%
from cmath import inf
import numpy as np
import pandas as pd
from javelin.lcmodel import Cont_Model
from javelin.zylc import get_data

n_iters = 90
# generate true light curve
# set parameters
for i in range(212,300):
    tau = 100  # signal decorrelation timescale in days
    sf = 0.2 # please keep ay this
    sigma_hat = sf/np.sqrt(tau)  # modified variability amplitude
    sigma = sf/np.sqrt(2)
    delta_t = 0.5  # observational cadence delta_t = t_i+1 - t_i in days
    duration = 4000  # in days
    mean_mag = 18 # please keep
    n_epochs = int(duration/delta_t)
    s = np.zeros(n_epochs)
    ts = np.zeros(n_epochs)

    for t in range(0, n_epochs):
        if t == 0:
            # for the 0th point, draw from Gaussian deviate of width sigma
            s[t] = np.random.normal(loc=0, scale=sigma)
            ts[t] = t*delta_t
            # print('Here! t = ' + str(t))
        else:
            s[t] = s[t-1]*np.exp(-delta_t/tau) + np.random.normal(loc=0,
                                                                scale=sigma*np.sqrt(1-np.exp(-2*delta_t/tau)))
            ts[t] = t*delta_t
            # print('Here! t = ' + str(t))

    # idk what the hell is happening here
    # sigma_sys = 0.004
    # sigma_rand = 
    # sigma_LSST = np.sqrt(sigma_sys**2 + sigma_rand**2)

    photometric_noise = np.random.normal(loc=0, scale=0.01, size=len(s))

    # light curve is actually s + Gaussian noise + mean magnitude
    y = s + mean_mag + photometric_noise

    # down sample light curve based on cadence
    c = 10
    ts_cadenced = ts[0::int(c/delta_t)]
    y_cadenced = y[0::int(c/delta_t)]
    photometric_noise_cadenced = photometric_noise[0::int(c/delta_t)]

    # final data to be saved
    ts_final = ts_cadenced
    y_final = y_cadenced
    photometric_noise_final = photometric_noise_cadenced

    the_data = np.zeros((len(ts_final),3))
    the_data[:,0] = ts_final
    the_data[:,1] = y_final
    the_data[:,2] = photometric_noise_final

    # MCMC saved light curves
    d = duration
    df = pd.DataFrame(the_data)
    df.to_csv(
        'DRW Results/Multi Curve/' + 'curve-d' +
        str(d) + '-c' + str(c) + '-iter' + str(i+1) + '.dat',
        header=False,
        index=False,
        sep=' ')

    javdata = get_data(
        ["DRW Results/Multi Curve/curve-d" +
            str(d) + "-c" + str(c) + '-iter' + str(i+1) + ".dat"],
        names=["$\\tau = 100$ days light curve", ])

    cont = Cont_Model(javdata)
    cont.do_mcmc(
        fchain="DRW Results/Multi Curve/" + "chain-d" +
            str(d) + "-c" + str(c) + '-iter' + str(i+1) +  ".dat",
        flogp="DRW Results/Multi Curve/logp" + "-d" +
            str(d) + "-c" + str(c) + '-iter' + str(i+1) +  ".dat",
        threads=8)

# %%
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from matplotlib import rc

### LaTeX Font for plotting ###
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
##########

iters = 87
medians = []
means = []

# works for d4000 c10
# for i in range(0,150):
#     if i == 20 or i == 87:
#     # if i == 30:
#         continue
#     else:
#         chain = "DRW Results/Multi Curve/chain-d4000-c10" + "-iter" + str(i + 1) +  ".dat"
#         df = pd.read_csv(chain,sep=' ',header=None)
#         medians.append(np.exp(np.median(df.values[:,1])))
#         means.append(np.exp(np.mean(df.values[:,1])))

#works for d1000c10
for i in range(0,148):
        chain = "DRW Results/Multi Curve/chain-d1000-c10" + "-iter" + str(i + 1) +  ".dat"
        df = pd.read_csv(chain,sep=' ',header=None)
        medians.append(np.exp(np.median(df.values[:,1])))
        means.append(np.exp(np.mean(df.values[:,1])))

plt.figure(figsize=(6,4))
n, bins, patches = plt.hist(
    medians,
    bins=20,
    label=r'Median $\tau$ distribution',
    # alpha=0.5,
    histtype='step',
    color='#eb811b')

# plt.hist(
#     means,
#     bins=20,
#     label=r'Mean $\tau$ distribution',
#     alpha=0.5)

plt.axvline(
    np.mean(medians),
    label=r'Average median $\tau$s',
    c = '#eb811b')
    # c='#23373b')

# plt.axvline(
#     np.mean(means),
#     label=r'Average mean $\tau$s',
#     c='#eb811b')

plt.axvline(
    100,
    label='Truth',
    c='tab:red')

plt.xlabel(
    r'$\tau$ (days)',
    fontsize=13)

plt.ylabel(
    'Frequency',
    fontsize=13)
    
plt.legend()
plt.title(
    'Median $\\tau$ ($n = {}$ quasars, 1000 days, 10-day cadence)'.format(len(means)),
    fontsize=14)

ax = plt.gca()
fig = plt.gcf()

dens_intervals = np.quantile(medians, (0.16, 0.5, 0.84))
med = np.mean(medians)

ax.text(
    110,
    16,
    r'$\tau = {:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$ days'.format(
        med,
        dens_intervals[2]-med,
        med-dens_intervals[0]),
    fontsize=13)

ax.tick_params(labelsize=12)

plt.savefig('Report Figures/multi-curve-hist-d1000-c10.pdf',dpi=300,bbox_inches='tight')

# print(np.mean(medians))
# print(np.mean(means))
 # %%
import scipy.stats as sts
import emcee as mc
import math

taus = np.asarray(medians)

def model(mu,sigma,alpha):
    return sts.skewnorm.logpdf(taus,alpha,mu,sigma) # (x,a,loc,scale)

def lnlike(params):
    mu, sigma, alpha = params
    p_tau_i = model(mu,sigma,alpha)
    # print(sum(p_tau_i))
    return sum(p_tau_i)

def lnprior(params):
    mu, sigma, alpha = params
    if sigma <= 0: # stop sigma = 0
        return -np.inf
    elif alpha < 0:
        return -np.inf
    return 0

def lnprob(params):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params)

nwalkers = 20
ndim = 3
burn_in = 3000

# mu around 70
# sigma around 40
# alpha around 4

vals = np.ones((nwalkers,ndim))
vals[:,0] = 70
vals[:,1] = 40
vals[:,2] = 4

p0 = vals + 10*np.random.rand(nwalkers,ndim)
sampler = mc.EnsembleSampler(nwalkers,ndim,lnprob,threads=8)
sampler.run_mcmc(p0,30000)

# %%
plt.figure()

samples = sampler.get_chain(flat=True)
sampled_no_burn = samples[burn_in:len(samples)]

plt.plot(samples[:,0], '.', label='mu walkers')
plt.plot(samples[:,1], '.', label='sigma walkers')
plt.plot(samples[:,2], '.', label='alpha walkers')
plt.legend()

plt.show()

plt.figure()
plt.plot(sampled_no_burn[:,0], '.', label='mu walkers')
plt.plot(sampled_no_burn[:,1], '.', label='sigma walkers')
plt.plot(sampled_no_burn[:,2], '.', label='alpha walkers')
plt.legend()

plt.show()

# %%
# draw corner plot
import corner
labels = ['$\\xi$','$\\omega$','$\\alpha$']

plt.figure()
corner.corner(
    sampled_no_burn,
    labels=labels,quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 16})
plt.savefig('corner-d1000-c10.jpg',dpi=300,transparent=False)
fig = plt.gcf()
fig.set_facecolor('white')
plt.show()

# %%
# overlay onto original hist
import matplotlib.patches as mpatches
plt.figure(figsize=(6, 4))

loc = np.median(sampled_no_burn[:,0])
scale = np.median(sampled_no_burn[:,1])
a = np.median(sampled_no_burn[:,2])
x = np.linspace(50,200,100)
y = sts.skewnorm.pdf(x,a,scale=scale,loc=loc)
y_max = np.max(y)
n_max = np.max(n)
scale_factor = n_max/y_max
y_scaled = y * scale_factor

plt.plot(
    x,
    y_scaled,
    label='Fitted skew-Gaussian',
    linestyle='-.',
    color='k')

loc = np.quantile(sampled_no_burn[:,0],[0.16,0.5,0.84])

plot_loc = (sampled_no_burn[:, 0])[
    (sampled_no_burn[:, 0] >= loc[0]) & (sampled_no_burn[:, 0] <= loc[2])]

inds = np.random.randint(len(plot_loc), size=100)

for i in inds:
    plt.plot(
        x,
        scale_factor * sts.skewnorm.pdf(x,a,scale=scale,loc=plot_loc[i]),
        alpha=0.02,
        color='tab:blue')

# plt.axvspan(
    # loc[0],
    # loc[2],
    # color='tab:blue',
    # alpha=0.5,
    # label='Range of $\\xi$')

iters = 87
medians = []
means = []
for i in range(0, 150):
    if i == 20 or i == 87:
        # if i == 30:
        continue
    else:
        chain = "DRW Results/Multi Curve/chain-d4000-c10" + \
            "-iter" + str(i + 1) + ".dat"
        df = pd.read_csv(chain, sep=' ', header=None)
        medians.append(np.exp(np.median(df.values[:, 1])))
        means.append(np.exp(np.mean(df.values[:, 1])))

plt.hist(
    medians,
    bins=20,
    label=r'Median $\tau$ distribution',
    # alpha=0.5,
    histtype='step',
    color='#eb811b')

plt.axvline(
    np.mean(medians),
    label=r'Average median $\tau$s',
    c='#eb811b')

plt.axvline(
    100,
    label='Truth',
    c='tab:red')

plt.xlabel(
    r'$\tau$ (days)',
    fontsize=13)

plt.ylabel(
    'Frequency',
    fontsize=13)

handles, labels = ax.get_legend_handles_labels()
blue_patch = mpatches.Patch(
    color='tab:blue', alpha=0.2, label='$\\Delta \\xi$')
handles.append(blue_patch)
plt.legend(handles=handles)

plt.title(
    'Median $\\tau$ ($n = {}$ quasars, 4000 days, 10-day cadence)'.format(len(means)),
    fontsize=14)

ax = plt.gca()
fig = plt.gcf()

dens_intervals = np.quantile(medians, (0.16, 0.5, 0.84))
med = np.mean(medians)

# ax.text(
    # 110,
    # 16,
    # r'$\tau = {:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$ days'.format(
        # med,
        # dens_intervals[2]-med,
        # med-dens_intervals[0]),
    # fontsize=13)

ax.tick_params(labelsize=12)
plt.savefig('fitted-dist.pdf',dpi=300,bbox_inches='tight')
plt.show()
# %%
