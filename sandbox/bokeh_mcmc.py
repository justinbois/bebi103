# Import Bokeh modules for interactive plotting
import bokeh.plotting

# DataShader
import datashader as ds
import datashader.bokeh_ext as ds_bokeh_ext

# Display graphics in this notebook
bokeh.io.output_notebook()

def data_range(df, cols, margin=0.02):
    x_range = df[cols[0]].max() - df[cols[0]].min()
    y_range = df[cols[1]].max() - df[cols[1]].min()
    return ([df[cols[0]].min() - x_range*margin, df[cols[0]].max()+ - x_range*margin],
            [df[cols[1]].min() - y_range*margin, df[cols[1]].max()+ - y_range*margin])

x_range, y_range = data_range(df_mcmc, ['r1', 'p1'])

def create_image(x_range, y_range, w, h):
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=int(h),
                    plot_width=int(w))
    agg = cvs.points(df_mcmc, 'r1', 'p1')
    img = ds.transfer_functions.shade(agg, cmap=ds.colors.viridis,
                                      how='linear')
#    img = ds.transfer_functions.set_background(img, ds.colors.viridis[0])
    return ds.transfer_functions.dynspread(img, threshold=0.8)

p = bokeh.plotting.figure(height=350, width=400, x_range=x_range, y_range=y_range,
                          x_axis_label='r1', y_axis_label='p1')
ds_bokeh_ext.InteractiveImage(p, create_image)

ds.transfer_functions.shade?

def hist_xy(data, bins=100):
    h, b = np.histogram(data, bins=bins, density=True)
    x = np.empty(len(b) * 2)
    y = np.empty(len(b) * 2)
    x[::2] = b
    x[1::2] = b
    y[1:-1:2] = r1
    y[2::2] = r1
    y[0] = 0
    y[-1] = 0
    return x, y

def data_range(df, cols, margin=0.02):
    x_range = df[cols[0]].max() - df[cols[0]].min()
    y_range = df[cols[1]].max() - df[cols[1]].min()
    return ([df[cols[0]].min() - x_range*margin, df[cols[0]].max()+ - x_range*margin],
            [df[cols[1]].min() - y_range*margin, df[cols[1]].max()+ - y_range*margin])


def two_d_slice(df, cols):
    if len(cols) != 2:
        raise RuntimeError('cols must be list of length 2')

    x_range, y_range = data_range(df, cols)




n = 3
p1 = bokeh.plotting.figure(plot_height=200, plot_width=200)
p1.line(*hist_xy(df_mcmc['r1'], bins=100), color='black')
p2 = bokeh.plotting.figure(plot_height=200, plot_width=200)
p2.line(*hist_xy(df_mcmc['p1'], bins=100), color='black')
p3 = bokeh.plotting.figure(plot_height=200, plot_width=200,
                           x_range=x_range, y_range=y_range,
                           x_axis_label='r1', y_axis_label='p1')
p1.x_range = p3.x_range
p2.x_range = p3.y_range
p1.yaxis.visible = False
p1.xaxis.visible = False
p2.yaxis.visible = False

grid = bokeh.layouts.gridplot([[p1, None], [p3, p2]])
bokeh.io.show(grid)
