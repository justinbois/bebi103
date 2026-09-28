.. _API:

API Reference
=============


Visualization
--------------------
.. currentmodule:: bebi103.viz

.. autosummary::
   :toctree: generated/viz
   :nosignatures:

   confints
   fill_between
   qqplot
   contour
   predictive_ecdf
   predictive_regression
   sbc_rank_ecdf
   parcoord
   trace
   corner
   contour_lines_from_samples
   cdf_to_staircase
   q_to_color
   mpl_cmap_to_color_mapper
   to_ColumnDataSource


Bootstrap methods
--------------------
.. currentmodule:: bebi103.bootstrap

.. autosummary::
   :toctree: generated/bootstrap
   :nosignatures:

   seed_rng
   draw_bs_reps
   draw_bs_reps_pairs
   draw_bs_reps_mle
   draw_perm_reps
   diff_of_means
   studentized_diff_of_means
   pearson_r


Hamiltonian Monte Carlo utilities
---------------------------------
.. currentmodule:: bebi103.hmc

.. autosummary::
   :toctree: generated/hmc
   :nosignatures:

   install_cmdstan_colab
   include_path
   clean_cmdstan
   cmdstan_version
   df_to_datadict_hier
   convert_samples
   extract_variable
   check_hmc_diagnostics
   check_expectand_diagnostics
   check_all_diagnostics
   tau_hat
   tail_xi_hat
   ess_hat
   split_rhat
   check_variance
   decode_warning_code
   disable_logging
   sample_cmdstanpy
   sample_cmdstanpy_prior_predictive
   parse_prior_predictive_samples
   inferential_calibration_cmdstanpy


Gaussian process utilities
-----------------------------
.. currentmodule:: bebi103.gp

.. autosummary::
   :toctree: generated/gp
   :nosignatures:

   append_sort_index
   linear_kernel
   polynomial_kernel
   se_kernel
   exp_quad_kernel
   matern_kernel
   periodic_kernel
   d1_se_kernel
   d1_exp_quad_kernel
   d2_se_kernel
   d2_exp_quad_kernel
   d1_d2_se_kernel
   d1_d2_exp_quad_kernel
   d1_matern_kernel
   d2_matern_kernel
   d1_d2_matern_kernel
   cov_exp_quad
   cov_se
   cov_d1_exp_quad
   cov_d1_se
   cov_d1_d2_exp_quad
   cov_d1_d2_se
   cov_matern
   cov_d1_matern
   cov_d1_d2_matern
   cov_periodic
   cov_from_kernel
   posterior_mean_cov
   posterior_mean_cov_deriv


HoloViews defaults
------------------
.. currentmodule:: bebi103.hv

.. autosummary::
   :toctree: generated/hv
   :nosignatures:

   set_defaults
   no_xgrid_hook
   no_ygrid_hook


Image processing utilities
-----------------------------
.. currentmodule:: bebi103.image

.. autosummary::
   :toctree: generated/image
   :nosignatures:

   imshow
   record_clicks
   draw_rois
   roicds_to_df
   im_merge
   rgb_to_rgba32
   rgb_frac_to_hex
   simple_image_collection
   verts_to_roi
   costes_coloc


General utilities
-----------------
.. currentmodule:: bebi103.utils

.. autosummary::
   :toctree: generated/utils
   :nosignatures:

   array_to_dict
   dict_to_array
   unpivot_csv
