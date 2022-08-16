import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# from this guy https://stackoverflow.com/questions/56715139/latex-table-with-uncertainty-from-pandas-dataframe
# from testing, this works as it should for errors
# i.e. with err_points = 1, errors are rounded to 1.s.f.
# which then sets the rounding for the actual mean value
def conv2siunitx(val, err, err_points=1):
    val = f'{val:.20e}'.split('e')
    err = f'{err:.20e}'.split('e') 
    first_uncertain = int(val[1]) - int(err[1]) + err_points

    my_val = f'{np.round(float(val[0]), first_uncertain-1):.10f}'
    my_err = f'{np.round(float(err[0]), err_points-1):.10f}'.replace('.','')
    # Avoid 1. and write 1 instead
    if first_uncertain > 1:
        first_uncertain = first_uncertain + 1

    return(f'{my_val[:first_uncertain]}({my_err[:err_points]})e{val[1]}')

def conv2PlusMinus(val,err):
    return str(val) + ' \pm ' + str(err)


# points_range = np.linspace(5*10**5, 10**7, num=11).astype(int)
# points_range = np.linspace(5*10**6, 20*10**6, num=16).astype(int)

# read in evidence csvs
# RESULTS1 10% uncertainty, obsPolar = (0.7, 4)
df1 = pd.read_csv('Results/1-08-r1/evidences.csv')
df2 = pd.read_csv('Results/1-08-r2/evidences.csv')
df3 = pd.read_csv('Results/1-08-r3/evidences.csv')
df4 = pd.read_csv('Results/4-08-r2/evidences.csv')
df5 = pd.read_csv('Results/5-08/evidences.csv')
df6 = pd.read_csv('Results/5-08-r2/evidences.csv')
df7 = pd.read_csv('Results/5-08-r3/evidences.csv')
df8 = pd.read_csv('Results/5-08-r4/evidences.csv')
df9 = pd.read_csv('Results/5-08-r5/evidences.csv')
df10 = pd.read_csv('Results/5-08-r6/evidences.csv')

# RESULTS2 (np.pi/2, 400), 10% uncertainty
# df1 = pd.read_csv('Results/7-08-r1/evidences.csv')
# df2 = pd.read_csv('Results/7-08-r2/evidences.csv')
# df3 = pd.read_csv('Results/7-08-r3/evidences.csv')
# df4 = pd.read_csv('Results/7-08-r4/evidences.csv')
# df5 = pd.read_csv('Results/7-08-r5/evidences.csv')

# 30% uncertainty, `obsPolar = (0.7,4)`
# df1 = pd.read_csv('Results/7-08-r6/evidences.csv')
# df2 = pd.read_csv('Results/8-08/evidences.csv')
# df3 = pd.read_csv('Results/8-08-r2/evidences.csv')
# df4 = pd.read_csv('Results/8-08-r3/evidences.csv')
# df5 = pd.read_csv('Results/8-08-r4/evidences.csv')

# obsPolar = (np.pi - 0.7, 4 - np.pi), 0.01%
points_range = np.linspace(10**6, 2*10**7, num=20).astype(int)
# df1 = pd.read_csv('Results/15-08/evidences-trials1sigma0.01.csv')
# df2 = pd.read_csv('Results/15-08/evidences-trials2sigma0.01.csv')
# df3 = pd.read_csv('Results/15-08/evidences-trials3sigma0.01.csv')
# df4 = pd.read_csv('Results/15-08/evidences-trials4sigma0.01.csv')
# df5 = pd.read_csv('Results/15-08/evidences-trials5sigma0.01.csv')
# df6 = pd.read_csv('Results/15-08/evidences-trials6sigma0.01.csv')
# df7 = pd.read_csv('Results/15-08/evidences-trials7sigma0.01.csv')

# df1 = pd.read_csv('Results/15-08/evidences-trials1sigma0.1.csv')
# df2 = pd.read_csv('Results/15-08/evidences-trials2sigma0.1.csv')
# df3 = pd.read_csv('Results/15-08/evidences-trials3sigma0.1.csv')
# df4 = pd.read_csv('Results/15-08/evidences-trials4sigma0.1.csv')
# df5 = pd.read_csv('Results/15-08/evidences-trials5sigma0.1.csv')

# df1 = pd.read_csv('Results/15-08/evidences-trials1sigma0.3.csv')
# df2 = pd.read_csv('Results/15-08/evidences-trials2sigma0.3.csv')
# df3 = pd.read_csv('Results/15-08/evidences-trials3sigma0.3.csv')
# df4 = pd.read_csv('Results/15-08/evidences-trials4sigma0.3.csv')
# df5 = pd.read_csv('Results/15-08/evidences-trials5sigma0.3.csv')

df1 = pd.read_csv('Results/15-08/evidences-trials1sigma0.5.csv')
df2 = pd.read_csv('Results/15-08/evidences-trials2sigma0.5.csv')
df3 = pd.read_csv('Results/15-08/evidences-trials3sigma0.5.csv')
df4 = pd.read_csv('Results/15-08/evidences-trials4sigma0.5.csv')
df5 = pd.read_csv('Results/15-08/evidences-trials5sigma0.5.csv')

# combine dataframes
total_frame = pd.concat([df1,df2,df3,df4,df5])
# print(total_frame)

# determine Bayes factor across trials
bayes_cmb_fitted = []
bayes_cmb_fitted_error = []
bayes_fitted_null = []
bayes_fitted_null_error = []
for i in range(0,len(df1['ln(Z)'])):
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
for i in range(0,len(df1['ln(Z)'])):
    means_fitted.append(np.mean((total_frame['ln(Z)'])[i])) # this selects lnZs that share an index i.e. across multiple trials
    fitted_evidence.append((total_frame['ln(Z)'])[i])
    devs_fitted.append(np.std((total_frame['ln(Z)'])[i]))


means_cmb = []
devs_cmb = []
cmb_evidence = []
for i in range(0,len(df1['ln(Z_CMB)'])):
    means_cmb.append(np.mean((total_frame['ln(Z_CMB)'])[i])) # this selects lnZs that share an index i.e. across multiple trials
    cmb_evidence.append((total_frame['ln(Z_CMB)'])[i])
    devs_cmb.append(np.std((total_frame['ln(Z_CMB)'])[i]))

means_null = []
devs_null = []
for i in range(0,len(df1['ln(Z_0)'])):
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
print('Fitted means:' + str(fitted))
print('Fitted errors:' + str(devs_fitted))

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
    np.asarray(errors_fitted.append(conv2siunitx(fitted[i],devs_fitted[i])))
    np.asarray(errors_null.append(conv2siunitx(null[i],devs_null[i])))
    np.asarray(errors_cmb.append(conv2siunitx(cmb[i], devs_cmb[i])))
    # np.asarray(errors_bayes.append(conv2siunitx(bayes_factor[i],bayes_factor_uncertainty[i])))

# print(errors_bayes)

nice2 = np.c_[points_range,errors_fitted,errors_cmb,errors_null,factor_cmb_fitted,factor_fitted_null]

results1 = (pd.DataFrame(nice2)) #instead of nice
# print(results1)
print(fitted_evidence)
print(cmb_evidence)

# plt.scatter(fitted_evidence[0],cmb_evidence[1])

bayes_factor = fitted_evidence[0]-cmb_evidence[0]
# print(bayes_factor)
# print(cmb_evidence)
# plt.scatter(cmb_evidence[0],bayes_factor)
plt.show()

# bayes factor vs log_cmb

with open('Results/15-08/table-s0.5.tex','w') as tex_file:
       tex_file.write(results1.to_latex(
        na_rep='',
        index=False,
        header = ['{$N$}',"{$\ln \mathcal{Z}$}","{$\ln \mathcal{Z}_{\\text{CMB}}$}",'{$\ln \mathcal{Z}_0$}','{$\ln \left( \mathcal{Z} / {\mathcal{Z}_{\\text{CMB}}}\\right)$}','{$\ln \left( \mathcal{Z} / {\mathcal{Z}_{0}}\\right)$}'],
        column_format='S[table-format=8.0,round-mode=none]S[table-format=3.2(1)e1]S[table-format=3.2(1)e1]S[table-format=3.2(1)e1]S[table-format=1(1),round-mode=uncertainty,round-precision=1]S[table-format=1(1),round-mode=uncertainty,round-precision=1]',
        escape=False))