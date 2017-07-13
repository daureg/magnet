import sys
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd

"""Read exp results and format them as latex table and text files"""


def load_res(prefix):
    """Concatenate all numpy arrays matching `prefix`."""
    res = []
    for fname in glob(prefix):
        with np.load(fname) as f:
            res.append(f['res'])
    return res[0] if len(res) == 1 else np.concatenate(res, 2)

metric = 2
prefix = sys.argv[1]
wres = load_res(prefix+'*')
v_mean = wres[:,:,:,metric].mean(-1).T
v_std = wres[:,:,:,metric].std(-1).T
x_vals = wres[0,:,:,-1].mean(-1).T
x_vals[-1]=1
col_name = ['\\blc$(t)$',  '\\blc$_{new}(t, u)$', '\\blc$_{old}(t, u)$',
            '\\blc$^\star(t)$', '\\blc$_{new}^\star(t, u)$',
            '\\blc$_{old}^\star(t, u)$', '\\logreg$^\star(t, u)$',
             '\\blc$_{new}^\star(a t, u)$']
x_ticks = ['${}\%$'.format(int(round(100*x, 0))) for x in x_vals]
index = x_ticks

def sfmt(val, rank, err, unit='', pm=True):
    start, end = '', ''
    if rank <= 3:
        start = ['\mathbf{1} -', '\mathbf{2} -', '\mathbf{3} -'][rank-1]
        start = ['', '', ''][rank-1]
        end = ''
    pms = '' if not pm else ' \pm {:.2f}'.format(err).replace('m 0.', 'm .')
    return '${}{:.2f}{}{}$ {}'.format(start, val, end, pms,  unit)

data = defaultdict(list)
skipped_column = None
# skipped_column = {1, }
for row_v, row_ve in zip(v_mean, v_std):
    for i, (v, ve, name) in enumerate(zip(row_v, row_ve, col_name)):
        if skipped_column and i in skipped_column:
            continue
        data[name].append(sfmt(100*v, 1, 100*ve))

index_name = '$\\frac{|E_{\mathrm{train}}|}{|E|}$'
name = '\{}{{}}'.format(prefix.split('_')[0])
# table = ["\\begin{table}\centering\caption{"+name+"}"]
table=['\setlength{\\tabcolsep}{4.5pt}', '\small']
df = pd.DataFrame(data, index)
table.append((df[col_name].to_latex()
             .replace('textbackslash', '').replace('\\textasciicircum', '^')
             .replace('\\$', '$').replace('\\_', '_').replace('\\{', '{')
             .replace('\\}', '}').replace('\\\\%', '\\%')
             .replace('{} &', index_name+' &').replace('lllllll', 'rccc|cccc')))
# table.append('\end{table}')
prefix += '_{}'.format({0: 'acc', 2: 'mcc'}[metric])
with open('tfull_{}.tex'.format(prefix), 'w') as f:
    f.write('\n'.join(table))

v_full = np.empty((v_mean.shape[0], v_mean.shape[1] + v_std.shape[1]+1),
                  dtype=v_mean.dtype)
v_full[:, 0] = 100*x_vals
v_full[:, 1::2] = v_mean
v_full[:, 2::2] = v_std
np.savetxt('vfull_{}'.format(prefix), v_full, '%.5f', delimiter="\t",
           header='\t'.join(['x', 'm1', 'm1_err', 'm2', 'm2_err', 'm3',
                             'm3_err', 'm4', 'm4_err', 'm5', 'm5_err',
                             'm6', 'm6_err', 'm7', 'm7_err', 'm8', 'm8_err']))
