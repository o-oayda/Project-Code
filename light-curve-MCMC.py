from cmath import inf
import numpy as np
import pandas as pd
from javelin.lcmodel import Cont_Model
from javelin.zylc import get_data
from funcs import window

n_iters = 300
duration = 4000  # in days
c = 10 # observational cadence in days
windowing = True
season_length = 90 # in days
season_gap = 90 # in days

# generate true light curve
for i in range(1,n_iters):
    tau = 100  # signal decorrelation timescale in days
    sf = 0.2  # structure function
    sigma_hat = sf/np.sqrt(tau)  # modified variability amplitude
    sigma = sf/np.sqrt(2)
    delta_t = 0.5  # intrinsic cadence delta_t = t_i+1 - t_i in days
    mean_mag = 18  # please keep
    n_epochs = int(duration/delta_t)
    s = np.zeros(n_epochs)
    ts = np.zeros(n_epochs)

    for t in range(0, n_epochs):
        if t == 0:
            # for the 0th point, draw from Gaussian deviate of width sigma
            s[t] = np.random.normal(loc=0, scale=sigma)
            ts[t] = t*delta_t
        else:
            s[t] = s[t-1]*np.exp(-delta_t/tau) + np.random.normal(
                loc=0,scale=sigma*np.sqrt(1-np.exp(-2*delta_t/tau)))
            ts[t] = t*delta_t

    photometric_noise = np.random.normal(loc=0, scale=0.01, size=len(s))

    # light curve points are s + Gaussian noise + mean magnitude
    y = s + mean_mag + photometric_noise

    # down sample light curve based on cadence
    ts_cadenced = ts[0::int(c/delta_t)]
    y_cadenced = y[0::int(c/delta_t)]
    photometric_noise_cadenced = photometric_noise[0::int(c/delta_t)]

    # apply windowing
    if windowing:
        ts_final = window(ts_cadenced,
            season_length=season_length,
            season_gap=season_gap,
            cadence=c)
        y_final = window(y_cadenced,
            season_length=season_length,
            season_gap=season_gap,
            cadence=c)
        photometric_noise_final = window(photometric_noise_cadenced,
            season_length=season_length,
            season_gap=season_gap,
            cadence=c)
    else:
        # unwindowed data
        ts_final = ts_cadenced
        y_final = y_cadenced
        photometric_noise_final = photometric_noise_cadenced

    # store into data matrix
    the_data = np.zeros((len(ts_final), 3))
    the_data[:, 0] = ts_final
    the_data[:, 1] = y_final
    the_data[:, 2] = photometric_noise_final

    # MCMC saved light curves
    d = duration
    df = pd.DataFrame(the_data)
    df.to_csv(
        'DRW Results/Multi Curve/curve-d' +\
            str(d) + '-c' + str(c) + '-iter' + str(i+1) + '.dat'\
            if not windowing else 'DRW Results/Multi Curve/cad-curve-d' +\
            str(d) + '-c' + str(c) + '-iter' + str(i+1) + '.dat',
        header=False,
        index=False,
        sep=' ')

    javdata = get_data(
        ["DRW Results/Multi Curve/curve-d" +\
            str(d) + "-c" + str(c) + '-iter' + str(i+1) + ".dat"]\
            if not windowing else ["DRW Results/Multi Curve/cad-curve-d" +\
            str(d) + "-c" + str(c) + '-iter' + str(i+1) + ".dat"],
        names=["$\\tau = 100$ days light curve", ])

    cont = Cont_Model(javdata)
    cont.do_mcmc(
        fchain="DRW Results/Multi Curve/" + "chain-d" +
            str(d) + "-c" + str(c) + '-iter' + str(i+1) + ".dat"\
            if not windowing else "DRW Results/Multi Curve/" + "cad-chain-d" +\
            str(d) + "-c" + str(c) + '-iter' + str(i+1) + ".dat",
        flogp="DRW Results/Multi Curve/logp" + "-d" +
            str(d) + "-c" + str(c) + '-iter' + str(i+1) + ".dat"\
            if not windowing else "DRW Results/Multi Curve/cad-logp" + "-d" +\
            str(d) + "-c" + str(c) + '-iter' + str(i+1) + ".dat",
        threads=8)