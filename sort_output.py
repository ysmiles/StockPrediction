import pandas as pd


filepath = 'data/'
filename = 'output_20000'
filepf = '.csv'  # file extension

d = pd.read_csv(filepath + filename + filepf)

col = [a.split('_') for a in d['Id']]
d = d.assign(c1=[int(a[0]) for a in col])
d = d.assign(c2=[int(a[1]) for a in col])

d.sort_values(by=['c1', 'c2'], inplace=True)

d.to_csv(filepath + filename + '_sorted' + filepf,
         index=False, columns=['Id', 'Predicted'])
