# Load DataFrame
df = pd.read_csv('../data/singer_transcript_counts.csv', comment='#')

# Look at ECDFs
def ecdf(data):
    return np.sort(data), np.arange(1, len(data)+1) / len(data)

def plot_ecdf(data, title=None):
    p = bokeh.plotting.figure(height=150, width=400, x_axis_label='transcript count',
                              y_axis_label='ECDF', title=title)
    p.circle(*ecdf(data))
    return p

plots = []
for i, col in enumerate(df.columns):
    plots.append(plot_ecdf(df[col], col))
bokeh.io.show(bokeh.layouts.gridplot([[p] for p in plots]))

def trace_to_df(trace):
    """This function would be in the bebi103 package. Need to update to account
    for 2D variables from the sampler and also to incorporate thinning."""
    variables = [var for var in trace.varnames if len(var) < 2 or var[-2:] != '__']
    cols = variables + ['chain']
    df = pd.DataFrame(columns=cols)
    for chain in trace.chains:
        data = np.array([trace.get_values(var) for var in variables]).transpose()
        data = np.concatenate((data, np.ones((len(data), 1)) * chain), axis=1)
        df = df.append(pd.DataFrame(data=data, columns=cols), ignore_index=True)
    df['chain'] = df['chain'].astype(int)
    
    return df
    
def sample_neg_binom(col, draws=2000, tune=500):
    with pm.Model() as neg_binom:
        # Prior on model parameters
        r = pm.Uniform('r', lower=0, upper=20)
        p = pm.Uniform('p', lower=0, upper=1)

        # Convert r, p to alpha, mu as expected from PyMC3's Negative Binomial
        alpha = r
        mu = r * (1/p - 1)

        # Likelihood
        n = pm.NegativeBinomial('n', alpha=alpha, mu=mu, observed=df[col].values)

        # Perform the sampling
        trace = pm.sample(draws=draws, tune=tune)

        return trace, neg_binom


def sample_double_neg_binom(col, draws=2000, tune=500):
    with pm.Model() as double_neg_binom:
        # Prior on model parameters
        r2 = pm.Uniform('r2', lower=0, upper=20)
        p2 = pm.Uniform('p2', lower=0, upper=1)
        r1 = pm.Uniform('r1', lower=0, upper=20)
        p1 = pm.Uniform('p1', lower=p2, upper=1)
        f = pm.Dirichlet('f', a=np.ones(2))

        # Convert r, p to alpha, mu as expected from PyMC3's Negative Binomial
        alpha1 = r1
        mu1 = r1 * (1/p1 - 1)
        alpha2 = r2
        mu2 = r2 * (1/p2 - 1)
                
        # Pieces of likelihood
        like_1 = pm.NegativeBinomial.dist(alpha=alpha1, mu=mu1)
        like_2 = pm.NegativeBinomial.dist(alpha=alpha2, mu=mu2)

        # Likelihood
        n = pm.Mixture('n', f, [like_1, like_2], observed=df[col].values)

        # Perform the sampling
        trace = pm.sample(draws=draws, tune=tune)

        return trace, double_neg_binom

pm.compare((trace_single, trace_double), (neg_binom, double_neg_binom))