.. _purpose:


Purpose and contents
====================

The BE/Bi 103 ab courses at Caltech cover basic data science principles such as data validation, wrangling, and visualization, as well as statistical inference. In some editions of the course, we also do image processing. 

For visualization, we use the `HoloViz <https://holoviz.org>`_ libraries, notably `Bokeh <http://bokeh.pydata.org/>`_, `HoloViews <http://holoviews.org/>`_, and `Panel <http://panel.holoviz.org/>`_. The statistical inference is done either using resampling (bootstrap) or using a Bayesian approach. For the latter, `Stan <http://mc-stan.org/>`_ is used heavily, with `ArviZ <https://arviz-devs.github.io/arviz/index.html>`_ being used to parse the results.

The bebi103 package consists of a set of utilities for facilitate the visualizations and analyses we use in the class, some of which extend or adapt the capabilities of the packages mentioned above. The package has five submodules.

- **hv**: Supplies colormaps and pretty default styling for HoloViews plotting elements.
- **viz**: Utilities for creating visualizations of data, but also of results from statistical inference calculations.
- **bootstrap**: Utilities to perform bootstrap-based statistical inference calculations. 
- **stan**: Utilities for using PyStan, CmdStanPy, and ArviZ for constructing, running, and parsing MCMC calculations with Stan.
- **gp**: Utilities for performing inference using Gaussian processes.
- **image**: Utilities for image processing applications.
