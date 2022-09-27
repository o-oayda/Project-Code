import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from funcs import conv2PlusMinus, conv2siunitx

test_index = input('Provide test index: ')

# read in evidence csvs
if test_index == 'A':
    # RESULTS1 10% uncertainty, obsPolar = (0.7, 4)
    test_files = [
        'Results/1-08-r1/evidences.csv',
        'Results/1-08-r2/evidences.csv',
        'Results/1-08-r3/evidences.csv',
        'Results/4-08-r2/evidences.csv',
        'Results/5-08/evidences.csv',
        'Results/5-08-r2/evidences.csv',
        'Results/5-08-r3/evidences.csv',
        'Results/5-08-r4/evidences.csv',
        'Results/5-08-r5/evidences.csv',
        'Results/5-08-r6/evidences.csv']

elif test_index == 'B':
    # RESULTS2 (np.pi/2, 4), 10% uncertainty
    test_files = [
        'Results/7-08-r1/evidences.csv',
        'Results/7-08-r2/evidences.csv',
        'Results/7-08-r3/evidences.csv',
        'Results/7-08-r4/evidences.csv',
        'Results/7-08-r5/evidences.csv']

elif test_index == 'C':
    # 30% uncertainty, `obsPolar = (0.7,4)`
    test_files = [
        'Results/7-08-r6/evidences.csv',
        'Results/8-08/evidences.csv',
        'Results/8-08-r2/evidences.csv',
        'Results/8-08-r3/evidences.csv',
        'Results/8-08-r4/evidences.csv',
        'Results/26-09/evidences-trials1sigma0.3.csv',
        'Results/26-09/evidences-trials2sigma0.3.csv', 
        'Results/26-09/evidences-trials3sigma0.3.csv',
        'Results/26-09/evidences-trials4sigma0.3.csv',
        'Results/26-09/evidences-trials5sigma0.3.csv',
        'Results/26-09/evidences-trials6sigma0.3.csv',
        'Results/26-09/evidences-trials7sigma0.3.csv',
        'Results/26-09/evidences-trials8sigma0.3.csv',
        'Results/26-09/evidences-trials9sigma0.3.csv',
        'Results/26-09/evidences-trials10sigma0.3.csv',
        'Results/26-09/evidences-trials11sigma0.3.csv',
        'Results/26-09/evidences-trials12sigma0.3.csv',
        'Results/26-09/evidences-trials13sigma0.3.csv',
        'Results/26-09/evidences-trials14sigma0.3.csv',
        'Results/26-09/evidences-trials15sigma0.3.csv',
        'Results/26-09/evidences-trials16sigma0.3.csv',
        'Results/26-09/evidences-trials17sigma0.3.csv',
        'Results/26-09/evidences-trials18sigma0.3.csv',
        'Results/26-09/evidences-trials19sigma0.3.csv']

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

elif test_index == 'E':
    test_files = [
        'Results/15-08/evidences-trials1sigma0.1.csv',
        'Results/15-08/evidences-trials2sigma0.1.csv',
        'Results/15-08/evidences-trials3sigma0.1.csv',
        'Results/15-08/evidences-trials4sigma0.1.csv',
        'Results/15-08/evidences-trials4sigma0.1.csv']

elif test_index == 'F':
    test_files = [
        'Results/15-08/evidences-trials1sigma0.3.csv',
        'Results/15-08/evidences-trials2sigma0.3.csv',
        'Results/15-08/evidences-trials3sigma0.3.csv',
        'Results/15-08/evidences-trials4sigma0.3.csv',
        'Results/15-08/evidences-trials4sigma0.3.csv']

elif test_index == 'G':
    test_files = [
        'Results/15-08/evidences-trials1sigma0.5.csv',
        'Results/15-08/evidences-trials2sigma0.5.csv',
        'Results/15-08/evidences-trials3sigma0.5.csv',
        'Results/15-08/evidences-trials4sigma0.5.csv',
        'Results/15-08/evidences-trials4sigma0.5.csv']

elif test_index == 'H':
    test_files = [
        'Results/22-08/evidences-trials1sigma0.1.csv',
        'Results/22-08/evidences-trials2sigma0.1.csv',
        'Results/22-08/evidences-trials3sigma0.1.csv',
        'Results/22-08/evidences-trials4sigma0.1.csv',
        'Results/22-08/evidences-trials5sigma0.1.csv']

elif test_index == 'I':
    test_files = [
        'Results/22-08/evidences-trials1sigma0.3.csv',
        'Results/22-08/evidences-trials2sigma0.3.csv',
        'Results/22-08/evidences-trials3sigma0.3.csv',
        'Results/22-08/evidences-trials4sigma0.3.csv',
        'Results/22-08/evidences-trials5sigma0.3.csv']

elif test_index == 'J':
    test_files = [
        'Results/22-08/evidences-trials1sigma0.5.csv',
        'Results/22-08/evidences-trials2sigma0.5.csv']

# combine dataframes
frames = []
for filename in test_files:
    df = pd.read_csv(filename)
    frames.append(df)

points_range = df['Points']
total_frame = pd.concat(frames)

# determine Bayes factor across trials
bayes_cmb_fitted = []
bayes_cmb_fitted_error = []
bayes_fitted_null = []
bayes_fitted_null_error = []
for i in range(0,len(points_range)):
    factors = total_frame['ln(Z)'][i] - total_frame['ln(Z_CMB)'][i]
    bayes_cmb_fitted.append(np.mean(factors))
    bayes_cmb_fitted_error.append(np.std(factors))
    factors2 = total_frame['ln(Z)'][i] - total_frame['ln(Z_0)'][i]
    bayes_fitted_null.append(np.mean(factors2))
    bayes_fitted_null_error.append(np.std(factors))

bayes_cmb_fitted = np.asarray(bayes_cmb_fitted,dtype=float)
bayes_cmb_fitted_error = np.asarray(bayes_cmb_fitted_error,dtype=float)
bayes_fitted_null = np.asarray(bayes_fitted_null,dtype=float)
bayes_fitted_null_error = np.asarray(bayes_fitted_null_error,dtype=float)

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
for i in range(0,len(points_range)):
    means_fitted.append(np.mean((total_frame['ln(Z)'])[i])) # this selects lnZs that share an index i.e. across multiple trials
    fitted_evidence.append((total_frame['ln(Z)'])[i])
    devs_fitted.append(np.std((total_frame['ln(Z)'])[i]))


means_cmb = []
devs_cmb = []
cmb_evidence = []
for i in range(0,len(points_range)):
    means_cmb.append(np.mean((total_frame['ln(Z_CMB)'])[i])) # this selects lnZs that share an index i.e. across multiple trials
    cmb_evidence.append((total_frame['ln(Z_CMB)'])[i])
    devs_cmb.append(np.std((total_frame['ln(Z_CMB)'])[i]))

means_null = []
devs_null = []
for i in range(0,len(points_range)):
    means_null.append(np.mean((total_frame['ln(Z_0)'])[i])) # this selects lnZs that share an index i.e. across multiple trials
    devs_null.append(np.std((total_frame['ln(Z_0)'])[i]))

fitted = np.asarray(means_fitted,dtype=float)
devs_fitted = np.asarray(devs_fitted,dtype=float)
cmb = np.asarray(means_cmb)
devs_cmb = np.asarray(devs_cmb)
null = np.asarray(means_null)
devs_null = np.asarray(devs_null)

nice = np.c_[points_range,fitted,cmb,fitted - cmb]
# print(nice)
print('CMB means:' + str(cmb))
print('CMB errors:' + str(devs_cmb))

# compute bayes factor and uncertainty by adding errors
# for two hypotheses in quadrature
# bayes_factor = fitted - cmb
# bayes_factor_uncertainty = np.zeros(len(fitted))
# print('Bayes factor: ' + str(bayes_factor))
# for i in range(0,len(fitted)):
#     bayes_factor_uncertainty[i] = np.sqrt(devs_fitted[i]**2 + devs_cmb[i]**2)
# print('Bayes factor uncertainties: ' + str(bayes_factor_uncertainty))

# use function to create string with errors
errors_fitted = []
errors_null = []
errors_cmb = []
# errors_bayes = []
for i in range(0,len(fitted)):
    # np.asarray(errors_fitted.append(conv2siunitx(fitted[i],devs_fitted[i])))
    # np.asarray(errors_null.append(conv2siunitx(null[i],devs_null[i])))
    # np.asarray(errors_cmb.append(conv2siunitx(cmb[i], devs_cmb[i])))
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

    # np.asarray(errors_bayes.append(conv2siunitx(bayes_factor[i],bayes_factor_uncertainty[i])))

# print(errors_bayes)

# dataframe to be exported to csv
nice2 = np.c_[points_range,errors_fitted,errors_cmb,errors_null,factor_cmb_fitted,factor_fitted_null]
results1 = (pd.DataFrame(nice2)) #instead of nice

# plotting results
## scatter of cmb evidence vs fitted evidence across number of trials for set number of points (index)
# plt.scatter(fitted_evidence[0],cmb_evidence[0])
# plt.show()

## scatter of cmb evidence vs bayes factor across number of trials for set number of points
# bayes_factor = fitted_evidence[0]-cmb_evidence[0]
# plt.scatter(cmb_evidence[0],bayes_factor)
# plt.show()

# bayes factor vs log_cmb

table_name = 'table' + str(test_index)
filename = str(table_name) + '-build'
with open('Results/' + str(table_name) + '.tex','w') as tex_file:
       tex_file.write(results1.to_latex(
        na_rep='',
        index=False,
        header = ['{$N$}',"{$\ln \mathcal{Z}$}","{$\ln \mathcal{Z}_{\\text{CMB}}$}",'{$\ln \mathcal{Z}_0$}','{$\ln \left( \mathcal{Z} / {\mathcal{Z}_{\\text{CMB}}}\\right)$}','{$\ln \left( \mathcal{Z} / {\mathcal{Z}_{0}}\\right)$}'],
        column_format='S[table-format=8,round-precision=0]S[table-format=6(3),round-mode=uncertainty,round-precision=1]S[table-format=6(3),round-mode=uncertainty,round-precision=1]S[table-format=6(3),round-mode=uncertainty,round-precision=1]S[table-format=1(1),round-mode=uncertainty,round-precision=1]S[table-format=1(1),round-mode=uncertainty,round-precision=1]',
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

plt.figure()

#  plot fitted to CMB Bayes factors
plt.errorbar(
    points_range,
    bayes_cmb_fitted,
    yerr=bayes_cmb_fitted_error,
    label=r'$\ln ( \mathcal{Z} / {\mathcal{Z}_{CMB}} )$',
    capsize=4,
    markersize=5,
    lw=1,
    fmt='o')

# plot fitted to null Bayes factors
plt.errorbar(
    points_range,
    bayes_fitted_null,
    yerr=bayes_fitted_null_error,
    label=r'$\ln ( \mathcal{Z} / \mathcal{Z}_0 )$',
    capsize=4,
    markersize=5,
    lw=1,
    fmt='o')

plt.xlabel('Number of Points')
plt.ylabel('Log Bayes Factor')
plt.grid(True)
plt.legend()
plt.title('Table ' + test_index + ' Bayes Factors')
plt.savefig('Results/' + 'table-' + test_index + '-plot.pdf')

plt.show()

# change to results directory and build standalone tex document
os.chdir('/Users/oliveroayda/Documents/GitHub/Project-Code/Results')
os.system('pdflatex ' + str(filename) + '.tex')