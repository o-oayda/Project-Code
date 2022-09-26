# %%
import numpy as np
import pandas as pd
from javelin.lcmodel import Cont_Model
from javelin.zylc import get_data

n_iters = 90
# generate true light curve
# set parameters
for i in range(31,50):
    tau = 100  # signal decorrelation timescale in days
    sf = 0.2
    sigma_hat = sf/np.sqrt(tau)  # modified variability amplitude
    sigma = sf/np.sqrt(2)
    delta_t = 0.5  # observational cadence delta_t = t_i+1 - t_i in days
    duration = 4000  # in days
    mean_mag = 18
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
    c = 5
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
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np

iters = 87
medians = []
means = []
for i in range(0,38):
    # if i == 20 or i == 87:
    if i == 30:
        continue
    else:
        chain = "DRW Results/Multi Curve/chain-d4000-c5" + "-iter" + str(i + 1) +  ".dat"
        df = pd.read_csv(chain,sep=' ',header=None)
        medians.append(np.exp(np.median(df.values[:,1])))
        means.append(np.exp(np.mean(df.values[:,1])))

figure(dpi=300)
plt.hist(medians,bins=15,label=r'Median $\tau$ distribution',alpha=0.5)
plt.hist(means,bins=15,label=r'Mean $\tau$ distribution',alpha=0.5)
plt.axvline(np.mean(medians),label=r'Average median $\tau$s',c='tab:blue')
plt.axvline(np.mean(means),label=r'Average mean $\tau$s',c='tab:orange')
plt.axvline(100,label='Truth',c='k')
plt.legend()
plt.title('Median/Mean $\\tau$ ($n = {}$ light curves, 4000 days, 5-day cadence)'.format(len(means)))

plt.show()


# plt.savefig('multi-curve-hist.pdf',dpi=300)

# print(np.mean(medians))
# print(np.mean(means))
# %%
