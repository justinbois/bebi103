.. _purpose:


Purpose and contents
====================

The BE/Bi 103 ab courses at Caltech cover basic data science principles such as data validation, wrangling, and visualization, as well as statistical inference. In some editions of the course, we also do image processing. 

For visualization, we use `Bokeh <https://bokeh.pydata.org/>`_ almost exclusively. Occasionally, though not often, we use `HoloViews <http://holoviews.org/>`_. The statistical inference is done either using resampling (bootstrap) or using a Bayesian approach. For the latter, `Stan <http://mc-stan.org/>`_ is used heavily, particularly the `CmdStanPy interface <https://cmdstanpy.readthedocs.io/>`_.

The bebi103 package consists of a set of utilities to facilitate the visualizations and analyses we use in the class, some of which extend or adapt the capabilities of the packages mentioned above. The package has six submodules.

- **viz**: Utilities for creating visualizations of data, but also of results from statistical inference calculations.
- **bootstrap**: Utilities to perform bootstrap-based statistical inference calculations. 
- **hmc**: Utilities for using Hamitonian Monte Carlo (HMC), primarily using CmdStanPy.
- **gp**: Utilities for performing inference using Gaussian processes.
- **hv**: Supplies a pretty default styling for HoloViews plotting elements.
- **image**: Utilities for image processing applications.
