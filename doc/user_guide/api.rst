.. _API:

API Reference
=============


HoloViews defaults
-------------------
.. currentmodule:: bebi103.hv

.. autosummary::
   :toctree: generated/hv
   :nosignatures:

   set_defaults
   no_xgrid_hook
   no_ygrid_hook


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


Stan utilities
--------------------
.. currentmodule:: bebi103.stan

.. autosummary::
   :toctree: generated/stan
   :nosignatures:

   clean_cmdstan
   df_to_datadict_hier
   posterior_to_dataframe
   check_divergences
   check_treedepth
   check_energy
   check_ess
   check_rhat
   check_all_diagnostics
   parse_warning_code
   sbc
   disable_logging


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
