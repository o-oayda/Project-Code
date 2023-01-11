from enum import unique
from re import A, L
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from funcs import conv2PlusMinus, conv2siunitx
import matplotlib.ticker
from matplotlib import rc
import matplotlib

 ### LaTeX Font for plotting ###
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
##########

# from https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
# for specifying the order of magnitude of the plot axis in scientific notation
class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

test_index = input('Provide test index: ')

# read in evidence csvs
if test_index == 'A':
    # RESULTS1 10% uncertainty, obsPolar = (0.7, 4)
    # test_files = [
    #     'Results/1-08-r1/evidences.csv',
    #     # 'Results/1-08-r2/evidences.csv', #annoying NaNs
    #     'Results/1-08-r3/evidences.csv',
    #     'Results/4-08-r2/evidences.csv',
    #     'Results/5-08/evidences.csv',
    #     'Results/5-08-r2/evidences.csv',
    #     'Results/5-08-r3/evidences.csv',
    #     'Results/5-08-r4/evidences.csv',
    #     'Results/5-08-r5/evidences.csv',
    #     'Results/5-08-r6/evidences.csv']
    test_files = []
    for i in range(1,21):
        test_files.append('Results/27-09/evidences-trials' + str(i) + 'sigma0.1.csv')
    
    # title = r'Bayes factors for $\Delta \tau_0 = 10 \%$ and $v = 0.001c$' + '\n' +  r'$30^\circ$ off-alignment with CMB'
    title = r'Bayes factors for $\Delta \tau_0 = 10 \%$ and $v = 0.001c$ (unaligned)'

elif test_index == 'B':
    # RESULTS2 (np.pi/2, 4), 10% uncertainty
    test_files = [
        'Results/7-08-r1/evidences.csv',
        'Results/7-08-r2/evidences.csv',
        'Results/7-08-r3/evidences.csv',
        'Results/7-08-r4/evidences.csv',
        'Results/7-08-r5/evidences.csv']
    
    title = r'Bayes factors: $\Delta \tau_0 = 10 \%; (\theta, \phi) = (\pi / 2,4)$'

elif test_index == 'C':
    # 30% uncertainty, `obsPolar = (0.7,4)`
    test_files = [
        'Results/7-08-r6/evidences.csv',
        'Results/8-08/evidences.csv',
        'Results/8-08-r2/evidences.csv',
        'Results/8-08-r3/evidences.csv',
        'Results/8-08-r4/evidences.csv']
        
    for i in range(1,20):
        test_files.append('Results/26-09/evidences-trials' + str(i) + 'sigma0.3.csv')
    
    title = r'Bayes factors: $\Delta \tau_0 = 30 \%; (\theta, \phi) = (0.7,4)$'

# obsPolar = (np.pi - 0.7, 4 - np.pi), 0.01%
elif test_index == 'D':
    test_files = [
        'Results/15-08/evidences-trials1sigma0.01.csv',
        'Results/15-08/evidences-trials2sigma0.01.csv',
        'Results/15-08/evidences-trials3sigma0.01.csv',
        'Results/15-08/evidences-trials4sigma0.01.csv',
        'Results/15-08/evidences-trials5sigma0.01.csv',
        'Results/15-08/evidences-trials6sigma0.01.csv',
        'Results/15-08/evidences-trials7sigma0.01.csv']
    
    title = r'Bayes factors: $\Delta \tau_0 = 1 \%; (\theta, \phi) = (2.4,0.9)$'

elif test_index == 'E':
    test_files = [
        'Results/15-08/evidences-trials1sigma0.1.csv',
        'Results/15-08/evidences-trials2sigma0.1.csv',
        'Results/15-08/evidences-trials3sigma0.1.csv',
        'Results/15-08/evidences-trials4sigma0.1.csv',
        'Results/15-08/evidences-trials4sigma0.1.csv']
    
    # title = r'Bayes factors: $\Delta \tau_0 = 10 \%; (\theta, \phi) = (\pi - 0.7,4 - pi)$'
    title = r'Bayes factors: $\Delta \tau_0 = 10 \%; (\theta, \phi) = (2.4,0.9)$'

elif test_index == 'F':
    test_files = [
        'Results/15-08/evidences-trials1sigma0.3.csv',
        'Results/15-08/evidences-trials2sigma0.3.csv',
        'Results/15-08/evidences-trials3sigma0.3.csv',
        'Results/15-08/evidences-trials4sigma0.3.csv',
        'Results/15-08/evidences-trials4sigma0.3.csv']

    for i in range(1,21):
        test_files.append('Results/28-09/evidences-trials' + str(i) + 'sigma0.3.csv')
    
    title = r'Bayes factors: $\Delta \tau_0 = 30 \%; (\theta, \phi) = (2.4,0.9)$'

elif test_index == 'G':
    test_files = [
        'Results/15-08/evidences-trials1sigma0.5.csv',
        'Results/15-08/evidences-trials2sigma0.5.csv',
        'Results/15-08/evidences-trials3sigma0.5.csv',
        'Results/15-08/evidences-trials4sigma0.5.csv',
        'Results/15-08/evidences-trials4sigma0.5.csv']

    for i in range(1,21):
        test_files.append('Results/29-09/evidences-trials' + str(i) + 'sigma0.5.csv')
    
    title = r'Bayes factors: $\Delta \tau_0 = 50 \%; (\theta, \phi) = (2.4,0.9)$'

elif test_index == 'H':
    test_files = [
        'Results/22-08/evidences-trials1sigma0.1.csv',
        'Results/22-08/evidences-trials2sigma0.1.csv',
        'Results/22-08/evidences-trials3sigma0.1.csv',
        'Results/22-08/evidences-trials4sigma0.1.csv',
        'Results/22-08/evidences-trials5sigma0.1.csv']
    
    title = r'Bayes factors: $\Delta \tau_0 = 10 \%; (\theta, \phi) = (1.04, \pi / 2)$'

elif test_index == 'I':
    test_files = [
        'Results/22-08/evidences-trials1sigma0.3.csv',
        'Results/22-08/evidences-trials2sigma0.3.csv',
        'Results/22-08/evidences-trials3sigma0.3.csv',
        'Results/22-08/evidences-trials4sigma0.3.csv',
        'Results/22-08/evidences-trials5sigma0.3.csv']

    for i in range(1,21):
        test_files.append('Results/29-09-v2/evidences-trials' + str(i) + 'sigma0.3.csv')
    
    title = r'Bayes factors: $\Delta \tau_0 = 30 \%; (\theta, \phi) = (1.04, \pi / 2)$'

elif test_index == 'J':
    test_files = [
        'Results/22-08/evidences-trials1sigma0.5.csv',
        'Results/22-08/evidences-trials2sigma0.5.csv']
    
    for i in range(1,21):
        test_files.append('Results/29-09-v2/evidences-trials' + str(i) + 'sigma0.5.csv')
    
    title = r'Bayes factors: $\Delta \tau_0 = 50 \%; (\theta, \phi) = (1.04, \pi / 2)$'

elif test_index == 'K':

    test_files = []
    for i in range(1,21):
        test_files.append('Results/23-10/evidences-trials{}sigma0.3.csv'.format(str(i)))

    title = r'Bayes factors: $\Delta \tau_0 = 30 \%; (\theta, \phi) = (0.7,4)$'

elif test_index == 'L':

    test_files = []
    for i in range(1, 20):
        test_files.append(
            'Results/29-12/evidences-trials{}sigma0.3.csv'.format(str(i)))

    for i in range(20,33):
        test_files.append(
            'Results/2-1/evidences-trials{}sigma0.3.csv'.format(str(i))
        )

    title = r'Bayes factors for $\Delta \tau_0 = 30 \%$ and $v=0.001c$ (unaligned)'

elif test_index == 'M':

    test_files = []
    for i in range(1, 21):
        test_files.append(
            'Results/30-12/evidences-trials{}sigma0.3.csv'.format(str(i)))

    title = r'Bayes factors for $\Delta \tau_0 = 30 \%$ and $v = 0.0024c$ (unaligned)'

elif test_index == 'N':
    # aligned with CMB
    # sigma 0.3
    # v = 0.0024c
    test_files = []
    for i in range(1,21):
        test_files.append(
            'Results/9-01/evidences-trials{}sigma0.3.csv'.format(str(i))
        )
    
    title = r'Bayes factors for $\Delta \tau_0 = 30 \%$ and $v = 0.0024c$ (aligned)'

##### COMPUTE BAYES FACTORS #######
# Read in csv filed containing evidences and combine into a dataframe
frames = []
for filename in test_files:
    df = pd.read_csv(filename)
    frames.append(df)

points_range = df['Points']
total_frame = pd.concat(frames)

# find unique points across trials
unique_points = sorted(list(set(total_frame['Points'])))

# extract all rows corresponding to a certain number of points
bayes_cmb_fitted = []
bayes_cmb_fitted_error = []
bayes_fitted_null = []
bayes_fitted_null_error = []
for k in unique_points:
    active_frame = total_frame.loc[total_frame['Points'] == k]
    factors = active_frame['ln(Z)'] - active_frame['ln(Z_CMB)']
    bayes_cmb_fitted.append(np.mean(factors))
    bayes_cmb_fitted_error.append(np.std(factors))
    factors2 = active_frame['ln(Z)'] - active_frame['ln(Z_0)']
    bayes_fitted_null.append(np.mean(factors2))
    bayes_fitted_null_error.append(np.std(factors))

bayes_cmb_fitted = np.asarray(bayes_cmb_fitted,dtype=float)
bayes_cmb_fitted_error = np.asarray(bayes_cmb_fitted_error,dtype=float)
bayes_fitted_null = np.asarray(bayes_fitted_null,dtype=float)
bayes_fitted_null_error = np.asarray(bayes_fitted_null_error,dtype=float)

# convert to string with the \pm symbol for LaTeX table
factor_cmb_fitted = []
factor_fitted_null = []
for i in range(0,len(bayes_cmb_fitted)):
    factor_cmb_fitted.append(conv2PlusMinus(
        np.asarray(bayes_cmb_fitted[i]),
        np.asarray(bayes_cmb_fitted_error[i])))
    factor_fitted_null.append(conv2PlusMinus(
        np.asarray(bayes_fitted_null[i]),
        np.asarray(bayes_fitted_null_error[i])))

# determine mean and 1 sigma across trials for fitted, null and CMB hypotheses
means_fitted = []
devs_fitted = []
fitted_evidence = []
for j in unique_points:
    active_frame = total_frame.loc[total_frame['Points'] == j]

    means_fitted.append(np.mean(active_frame['ln(Z)']))
    fitted_evidence.append(active_frame['ln(Z)'])
    devs_fitted.append(np.std(active_frame['ln(Z)']))

means_cmb = []
devs_cmb = []
cmb_evidence = []
for j in unique_points:
    active_frame = total_frame.loc[total_frame['Points'] == j]

    means_cmb.append(np.mean(active_frame['ln(Z_CMB)']))
    cmb_evidence.append(active_frame['ln(Z_CMB)'])
    devs_cmb.append(np.std(active_frame['ln(Z_CMB)']))

means_null = []
devs_null = []
for j in unique_points:
    active_frame = total_frame.loc[total_frame['Points'] == j]

    means_null.append(np.mean(active_frame['ln(Z_0)']))
    devs_null.append(np.std(active_frame['ln(Z_0)']))

fitted = np.asarray(means_fitted,dtype=float)
devs_fitted = np.asarray(devs_fitted,dtype=float)
cmb = np.asarray(means_cmb)
devs_cmb = np.asarray(devs_cmb)
null = np.asarray(means_null)
devs_null = np.asarray(devs_null)

# use conv2PlusMinus to create string with \pm symbol as errors
errors_fitted = []
errors_null = []
errors_cmb = []
for i in range(0,len(fitted)):
    errors_fitted.append(
        conv2PlusMinus(
            np.asarray(fitted[i]),
            np.asarray(devs_fitted[i])))
    
    errors_null.append(
        conv2PlusMinus(
            np.asarray(null[i]),
            np.asarray(devs_null[i])))

    errors_cmb.append(
        conv2PlusMinus(
            np.asarray(cmb[i]),
            np.asarray(devs_cmb[i])))

# dataframe to be exported to csv --- replaced with unique points
export = np.c_[unique_points,errors_fitted,errors_cmb,errors_null,factor_cmb_fitted,factor_fitted_null]
results1 = (pd.DataFrame(export))

# write all evidence values and Bayes factors to .tex table
table_name = 'table' + str(test_index)
filename = str(table_name) + '-build'
with open('Results/' + str(table_name) + '.tex','w') as tex_file:
       tex_file.write(results1.to_latex(
        na_rep='',
        index=False,
        header = ['{$N$}',
        "{$\ln \mathcal{Z}$}",
        "{$\ln \mathcal{Z}_{\\text{CMB}}$}",
        '{$\ln \mathcal{Z}_0$}',
        '{$\ln \left( \mathcal{Z} / {\mathcal{Z}_{\\text{CMB}}}\\right)$}',
        '{$\ln \left( \mathcal{Z} / {\mathcal{Z}_{0}}\\right)$}'],
        column_format='''S[table-format=8,round-precision=0]
            S[table-format=6(3),round-mode=uncertainty,round-precision=1]
            S[table-format=6(3),round-mode=uncertainty,round-precision=1]
            S[table-format=6(3),round-mode=uncertainty,round-precision=1]
            S[table-format=1(1),round-mode=uncertainty,round-precision=1]
            S[table-format=1(1),round-mode=uncertainty,round-precision=1]''',
        escape=False))

# put table in standalone tex document
with open("Results/" + str(filename) + ".tex", "w") as tex_file:
    tex_file.write(r"""
\documentclass[varwidth=\maxdimen]{standalone}
% \documentclass{memoir}
\usepackage{booktabs}
\usepackage[table, svgnames]{xcolor}
\usepackage[round-mode=places,separate-uncertainty=true,table-align-uncertainty=true]{siunitx}
\setlength{\extrarowheight}{1mm}

\rowcolors{1}{black!10}{}
\setlength{\aboverulesep}{0pt}
\setlength{\belowrulesep}{0pt}

\begin{document}

\begin{table}
    \input{""" + str(table_name) + """}
\end{table}

\end{document}""")

plt.figure(figsize=(6.4,6))

#  plot fitted to CMB Bayes factors
plt.errorbar(
    unique_points,
    bayes_cmb_fitted,
    yerr=bayes_cmb_fitted_error,
    # label=r'$\ln B_{12} = \ln ( \mathcal{Z} / {\mathcal{Z}_{CMB}} )$',
    label=r'$\ln B_{12}$ (fitted vs.\! CMB hypothesis)',
    capsize=4,
    markersize=4,
    lw=1,
    fmt='o',
    c='#eb811b') # met orange

# plot fitted to null Bayes factors
plt.errorbar(
    unique_points,
    bayes_fitted_null,
    yerr=bayes_fitted_null_error,
    # label=r'$\ln B_{10} = \ln ( \mathcal{Z} / \mathcal{Z}_0 )$',
    label=r'$\ln B_{10}$ (fitted vs.\! null hypothesis)',
    capsize=4,
    markersize=4,
    lw=1,
    fmt='o',
    c='#23373b') # met blue

plt.xlabel('Number of Sources',fontsize=18)
plt.ylabel('Log Bayes Factor',fontsize=18)
plt.grid(True)
plt.legend(fontsize=14)
plt.title(title,fontsize=18)
plt.ticklabel_format(useMathText=True)

ax = plt.gca()
fig = plt.gcf()

ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.1f"))

ax.tick_params(labelsize=14)
ax.xaxis.offsetText.set_fontsize(14)
plt.savefig('Results/' + 'table-' + test_index + '-plot.pdf',bbox_inches='tight')

plt.show()

# change to results directory and build standalone tex document
os.chdir('/Users/oliveroayda/Documents/GitHub/Project-Code/Results')
os.system('pdflatex ' + str(filename) + '.tex')