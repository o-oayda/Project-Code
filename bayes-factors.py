import pandas as pd
import numpy as np
import os

# from this guy https://stackoverflow.com/questions/56715139/latex-table-with-uncertainty-from-pandas-dataframe
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

points_range = np.linspace(5*10**5, 10**7, num=11).astype(int)

# read in evidence csvs
df1 = pd.read_csv('Results/1-08-r1/evidences.csv')
df2 = pd.read_csv('Results/1-08-r2/evidences.csv')
df3 = pd.read_csv('Results/1-08-r3/evidences.csv')

# combine dataframes
total_frame = pd.concat([df1,df2,df3])
print(total_frame)

# determine mean and 1 sigma across trials for fitted, null and CMB hypotheses
means_fitted = []
devs_fitted = []
for i in range(0,len(df1['ln(Z)'])):
    means_fitted.append(np.mean((total_frame['ln(Z)'])[i])) # this selects lnZs that share an index i.e. across multiple trials
    devs_fitted.append(np.std((total_frame['ln(Z)'])[i]))

means_cmb = []
devs_cmb = []
for i in range(0,len(df1['ln(Z_CMB)'])):
    means_cmb.append(np.mean((total_frame['ln(Z_CMB)'])[i])) # this selects lnZs that share an index i.e. across multiple trials
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

# use function to create string with errors
errors_fitted = []
errors_null = []
errors_cmb = []
for i in range(0,len(fitted)):
    np.asarray(errors_fitted.append(conv2siunitx(fitted[i],devs_fitted[i])))
    np.asarray(errors_null.append(conv2siunitx(null[i],devs_null[i])))
    np.asarray(errors_cmb.append(conv2siunitx(cmb[i], devs_cmb[i])))

nice2 = np.c_[points_range,errors_fitted,errors_cmb,fitted - cmb]

results1 = (pd.DataFrame(nice2)) #instead of nice
print(results1)

with open('Results/table.tex','w') as tex_file:
       tex_file.write(results1.to_latex(
        na_rep='',
        index=False,
        header = ['{Points}',"{Fitted evidence}","{CMB evidence}",'{Bayes factor}'],
        column_format='S[table-format=1.0]SSS',
        escape=False))