# %%
from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model

tau = 100 # true light curve time scale
d = 40*tau # curve duration (days)
c = 5 # curve cadence (days)

javdata = get_data(
    ["DRW Results/curve-d" + str(d) + "-c" + str(c) + ".dat"],
    names=["$\\tau = 100$ days light curve", ])

iters = 20

for i in range(0,iters):
    cont = Cont_Model(javdata)
    cont.do_mcmc(
        fchain="DRW Results/Consistency Check/" + "chain-d" + str(d) + "-c" + str(c) + '-iter' + str(i+1) + ".dat",
        flogp="DRW Results/logp" + "-d" + str(d) + "-c" + str(c)+ '-iter' + str(i+1) + ".dat",
        threads=8)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iters = 20
medians = []
means = []
for i in range(0,iters):
    chain = "DRW Results/Consistency Check/chain-d4000-c5" + "-iter" + str(i + 1) +  ".dat"
    df = pd.read_csv(chain,sep=' ',header=None)
    medians.append(np.exp(np.median(df.values[:,1])))
    means.append(np.exp(np.mean(df.values[:,1])))

print(medians)
print(means)
plt.hist(medians,bins=20)
plt.hist(means,bins=20)
# %%
