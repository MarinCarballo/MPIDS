import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from statistics import stdev
from statistics import mean
import matplotlib.pyplot as plt

#modify file_name to be the same as the file with the data duhhh
#make sure the csv file in in the same folder as this code.
file_name = "Ep-Nu.csv"

#order names in the same order as the regression that you are interested, the name don't
#have to be the same as the ones you have for the header in your csv file.
#It's recommended to leave the naming as x, y, dx, dy to not have to make further modifications
#to the rest of the code.
name = ['x', 'dx', 'y', 'dy']
data = pd.read_csv(file_name, delimiter=',', header = 0, names = name)

#Function to adjust to:
def func(x, a, b):
    return a*x+b

popt, pcov = curve_fit(func, data.x, data.y, sigma=data.dy, absolute_sigma=True)

#Finding the linear regression parameters
y_pred = func(data.x, *popt)
r = data.y - y_pred
chisq = sum((r / data.dy) ** 2)
mediax = mean(data.x)
mediay = mean(data.y)

mediaxy = mean(data.x*data.y)
sigmax = stdev(data.x)
sigmay = stdev(data.y)
coef_Pearson = len(data.x)*(mediaxy-mediax*mediay)/((len(data.x)-1)*sigmax*sigmay)

#output of the results

print('Fit values:')
print(f'   a: {popt[0]} \u00B1 {np.sqrt(pcov.diagonal())[0]}')
if popt.size > 1:
    print(f'   b: {popt[1]} \u00B1 {np.sqrt(pcov.diagonal())[1]}')

print('')
print(f'chi\u00b2: {chisq}')
print(f'Pearson: {coef_Pearson}')

fig=plt.figure(figsize=[10,6])
ax=fig.add_subplot(111)
plt.errorbar(data.x, data.y, xerr=data.dx, yerr=data.dy, fmt='.', label='Data', linewidth=2)
plt.plot(data.x, y_pred, 'r-', label='Fit',linewidth=1)
plt.xlabel('Ns',fontsize=12)
plt.ylabel('ε (V)',fontsize=12)
plt.legend(loc='best',fontsize=12)
plt.grid()