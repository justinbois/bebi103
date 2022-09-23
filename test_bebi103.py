import numpy as np
import pandas as pd

import bokeh.plotting

import bebi103


def test_bokeh_boxplot():
    labels = list('ADBACBABDABDBDBCBABCBADBADBBADBACBADBAD')
    values = np.array([1.67,  0.23,  2.51,  1.06,  0.22,
                      -0.54,  1.07,  0.07, -0.15, -1.47,
                       1.42, -0.28, -0.38, -1.24, -0.78,
                      -0.25, -0.49,  0.78,  0.25,  0.81,
                      -1.20, -0.53, -1.33, -0.36, -0.63,
                       0.74, -0.77, -0.68,  0.04,  0.32,
                       0.35,  0.90, -2.91,  1.04,  0.86,
                      -0.76,  0.56,  1.98, -1.13])
    df = pd.DataFrame({'label': labels, 'value': values})
    bokeh.plotting.output_file('test_boxplot.html')
    p = bebi103.bokeh_boxplot(df, value='value', label='label', sort=False)
    bokeh.plotting.show(p)
    print('Medians:')
    print(df.groupby('label').median())
    y_n = input('Visual inspection ok? ')
    if y_n in ['y', 'Y', '1']:
        return True
    else:
        return False


def test_bokeh_matplot():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, None]]).astype(float)
    a[1, 1] = None
    data = np.array(np.unravel_index(range(9), a.shape) + (a.ravel(),)).T
    df = pd.DataFrame(data, columns=['i', 'j', 'data'])

    bokeh.plotting.output_file('test_matplot.html')
    # Numerical x-axis
    p_numerical = bebi103.bokeh_matplot(df, 'i', 'j', 'data', n_colors=9,
                                        colormap='RdBu_r', plot_width=200,
                                        plot_height=200)

    # Categorical x-axis, i.e. column indices (labels are strings)
    df.j = df.j.astype(str)
    p_categorical = bebi103.bokeh_matplot(df, 'i', 'j', 'data', n_colors=9,
                                          colormap='RdBu_r', plot_width=200,
                                          plot_height=200)

    bokeh.plotting.show(bokeh.io.vplot(p_categorical, p_numerical))
    y_n = input('Visual inspection ok? (X-Axis -> Top: above, Bottom: below)')
    if y_n in ['y', 'Y', '1']:
        return True
    else:
        return False


# To go in run_ensemble_sampler:
# assert np.allclose(sampler.chain[0,:,0], df[df.chain==0][df.columns[0]])
# assert np.allclose(sampler.chain[1,:,0], df[df.chain==1][df.columns[0]])
# assert np.allclose(sampler.chain[13,:,1], df[df.chain==13][df.columns[1]])
# assert np.allclose(sampler.chain[45,:,1], df[df.chain==45][df.columns[1]])

# To go in run_pt_sampler:
# inds = (df['chain'] == 0) & (df['beta_ind'] == 2)
# assert np.allclose(sampler.chain[2, 0, :, 0], df[inds][df.columns[0]])
# assert np.allclose(sampler.lnprobability[2, 0], df[inds]['lnprob'])
