# %%
import matplotlib.patches as mpatches
import corner
import math
import emcee as mc
import scipy.stats as sts
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from matplotlib import rc
import multiprocess

### LaTeX Font for plotting ###
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
##########

d = input('Provide duration (days): ')
c = input('Provide cadence (days): ')
n_quasars = 300 # number of collected quasars
# windowing = ('Windowed (T/F)?')

medians = []
means = []
for i in range(n_quasars):
    chain = "DRW Results/Multi Curve/cad-chain-d" + str(d) + "-c" + str(c) + \
        "-iter" + str(i + 1) + ".dat"
    df = pd.read_csv(chain, sep=' ', header=None)
    medians.append(np.exp(np.median(df.values[:, 1])))
    means.append(np.exp(np.mean(df.values[:, 1])))

plt.figure(figsize=(6, 4))
n, bins, patches = plt.hist(
    medians,
    bins=15,
    label=r'$\tau$ distribution',
    histtype='step',
    color='#eb811b')

plt.axvline(
    np.median(medians),
    label=r'Median of distribution',
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

plt.legend()
plt.title(
    '$\\tau$ Distribution ($n = {}$, $t={}$ days, $\\Delta t = {}$ days, windowed)'.format(
        len(means),str(d),str(c)),
    fontsize=14)

ax = plt.gca()
fig = plt.gcf()

dens_intervals = np.quantile(medians, (0.16, 0.5, 0.84))
med = np.mean(medians)

ax.text(
    0.95,
    0.75,
    r'$\tau = {:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$ days'.format(
        dens_intervals[1],
        dens_intervals[2]-dens_intervals[1],
        dens_intervals[1]-dens_intervals[0]),
    horizontalalignment='right',
    verticalalignment='top',
    transform = ax.transAxes,
    fontsize=13)

ax.tick_params(labelsize=12)

plt.savefig('Report Figures/cad-multi-curve-hist-d' + str(d) + '-c' + str(c) + '.pdf',
            dpi=300, bbox_inches='tight')

# %%
# fit skewed Gaussian with MCMC
import dynesty
from dynesty import plotting as dyplot

method = input('Select method: ')
taus = np.asarray(medians)
labels = ['$\\xi$', '$\\omega$', '$\\alpha$']

def model(mu, sigma, alpha):
    return sts.skewnorm.logpdf(taus, alpha, mu, sigma)  # (x,a,loc,scale)

def lnlike(params):
    mu, sigma, alpha = params
    p_tau_i = model(mu, sigma, alpha)
    # print(sum(p_tau_i))
    return sum(p_tau_i)

def lnprior(params):
    mu, sigma, alpha = params
    if sigma <= 0:  # stop sigma = 0
        return -np.inf
    elif alpha < 0:
        return -np.inf
    return 0

def prior_transform(uTheta):
    uMu, uSigma, uAlpha = uTheta
    mu = 100 * uMu
    sigma = 200 * uSigma
    alpha = 10 * uAlpha
    return mu, sigma, alpha

def lnprob(params):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params)

if method == 'emcee':
    nwalkers = 10
    ndim = 3
    burn_in = 10000

    # mu around 70
    # sigma around 40
    # alpha around 4
    vals = np.ones((nwalkers, ndim))
    vals[:, 0] = 70
    vals[:, 1] = 40
    vals[:, 2] = 4

    p0 = vals + 5*np.random.rand(nwalkers, ndim)
    sampler = mc.EnsembleSampler(nwalkers, ndim, lnprob, threads=8)
    sampler.run_mcmc(p0, 20000)

    plt.figure()
    samples = sampler.get_chain(flat=True)
    sampled_no_burn = samples[burn_in:len(samples)]
    plt.plot(samples[:, 0], '.', label='mu walkers')
    plt.plot(samples[:, 1], '.', label='sigma walkers')
    plt.plot(samples[:, 2], '.', label='alpha walkers')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(sampled_no_burn[:, 0], '.', label='mu walkers')
    plt.plot(sampled_no_burn[:, 1], '.', label='sigma walkers')
    plt.plot(sampled_no_burn[:, 2], '.', label='alpha walkers')
    plt.legend()
    plt.show()

    plt.figure()
    corner.corner(
        sampled_no_burn,
        labels=labels, quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 17},
        label_kwargs={"fontsize": 15})

    fig = plt.gcf()
    ax = plt.gca()
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=13)
    plt.savefig(
        'Report Figures/cad-corner-d{}-c{}.pdf'.format(str(d), str(c)),
        dpi=300,
        bbox_inches='tight')
    plt.show()

elif method == 'dynesty':
    with multiprocess.Pool(8) as pool:
        dsampler = dynesty.DynamicNestedSampler(
            lnlike,
            prior_transform,
            ndim=3,
            pool=pool,
            queue_size=8)
        dsampler.run_nested()
        dresults = dsampler.results

        fig, axes = dyplot.cornerplot(
            dresults,
            show_titles=True,
            title_kwargs={'y': 1.03, 'fontsize' : 18},
            label_kwargs={'fontsize' : 17},
            labels=labels,
            title_fmt='.2f',
            quantiles=[0.16,0.5,0.84],
            title_quantiles=[0.16,0.5,0.84])
        
        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=13)
        
        plt.savefig(
            'Report Figures/dycorner-d{}-c{}.pdf'.format(str(d), str(c)),
            dpi=300)

else:
    print('Invalid method.')

# %%
# overlay onto original hist

if method == 'emcee':

    plt.figure(figsize=(6, 4))

    loc = np.median(sampled_no_burn[:, 0])
    scale = np.median(sampled_no_burn[:, 1])
    a = np.median(sampled_no_burn[:, 2])
    x = np.linspace(min(medians), max(medians), 100)
    y = sts.skewnorm.pdf(x, a, scale=scale, loc=loc)
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

    loc = np.quantile(sampled_no_burn[:, 0], [0.16, 0.5, 0.84])

    plot_loc = (sampled_no_burn[:, 0])[
        (sampled_no_burn[:, 0] >= loc[0]) & (sampled_no_burn[:, 0] <= loc[2])]

    inds = np.random.randint(len(plot_loc), size=100)

    for i in inds:
        plt.plot(
            x,
            scale_factor * sts.skewnorm.pdf(x, a, scale=scale, loc=plot_loc[i]),
            alpha=0.02,
            color='tab:blue')

    plt.hist(
        medians,
        bins=15,
        label=r'$\tau$ distribution',
        histtype='step',
        color='#eb811b')

    plt.axvline(
        np.median(medians),
        label=r'Median of distribution',
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

    ax = plt.gca()

    handles, labels = ax.get_legend_handles_labels()
    blue_patch = mpatches.Patch(
        color='tab:blue',
        alpha=0.2,
        label=r'$\Delta \xi ={: .2f} ^ {{+{: .2f}}}_{{-{: .2f}}}$'.format(
            loc[1],
            loc[2]-loc[1],
            loc[1]-loc[0]))
    handles.append(blue_patch)
    plt.legend(handles=handles)

    plt.title(
        '$\\tau$ Distribution ($n = {}$, $t={}$ days, $\\Delta t = {}$ days)'.format(
            len(means),str(d),str(c)),
        fontsize=14)

    ax.text(
    0.95,
    0.6,
    r'$\tau = {:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$ days'.format(
        dens_intervals[1],
        dens_intervals[2]-dens_intervals[1],
        dens_intervals[1]-dens_intervals[0]),
    horizontalalignment='right',
    verticalalignment='top',
    transform = ax.transAxes,
    fontsize=13)

    ax.tick_params(labelsize=12)
    plt.savefig(
        'Report Figures/cad-fitted-dist-d{}-c{}.pdf'.format(str(d),str(c)),
        dpi=300,
        bbox_inches='tight')
    plt.show()

# %%
