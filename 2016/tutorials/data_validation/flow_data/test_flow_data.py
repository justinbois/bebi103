import glob

import numpy as np
import pandas as pd

fnames = glob.glob('*RBS1027*.csv')

def test_column_names():
    """Make sure all data sets have proper column names."""
    column_names = ['FSC-A', 'SSC-A', 'FITC-A', 'gate']
    for fname in fnames:
        df = pd.read_csv(fname)
        assert list(df.columns) == column_names, \
                    fname + ' has wrong column names.'

def test_missing_data():
    """Look for missing entries."""
    for fname in fnames:
        df = pd.read_csv(fname)
        assert np.all(df.notnull()), fname + ' contains missing data'


def test_negative_scattering():
    """Look for negative scattering values"""
    for fname in fnames:
        df = pd.read_csv(fname)
        assert np.all(df.loc[df.gate==1, ['FSC-A', 'SSC-A', 'FITC-A']] >= 0), \
            fname + ' contains negative scattering data'
