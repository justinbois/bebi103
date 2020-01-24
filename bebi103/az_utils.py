"""This submodule contains modifications of functions
included in ArviZ's plots.plot_utils submodule. I include them here with
slight modifications because of a bug in the purge_duplicates() function
in ArviZ version 0.6.1. This bug is addressed in this pull request:
https://github.com/arviz-devs/arviz/pull/1027. Until that is merged,
this submodule is used in bebi103 to convert xarray data to Numpy 
arrays.

The code is licensed under an Apache 2.0 License, reproduced below.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "{}"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright {yyyy} {name of copyright owner}

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import itertools
import numpy as np
import xarray as xr


def _dims(data, var_name, skip_dims):
    return [dim for dim in data[var_name].dims if dim not in skip_dims]


def _zip_dims(new_dims, vals):
    return [{k: v for k, v in zip(new_dims, prod)} for prod in itertools.product(*vals)]


def purge_duplicates(list_in):
    """Remove duplicates from list while preserving order.

    Parameters
    ----------
    list_in: Iterable

    Returns
    -------
    list
        List of first occurrences in order
    """
    # Algorithm taken from Stack Overflow,
    # https://stackoverflow.com/questions/480214. Content by Georgy
    # Skorobogatov (https://stackoverflow.com/users/7851470/georgy) and
    # Markus Jarderot
    # (https://stackoverflow.com/users/22364/markus-jarderot), licensed
    # under CC-BY-SA 4.0.
    # https://creativecommons.org/licenses/by-sa/4.0/.

    seen = set()
    seen_add = seen.add
    return [x for x in list_in if not (x in seen or seen_add(x))]


def make_label(var_name, selection, position="below"):
    """Consistent labelling for plots.
    Parameters
    ----------
    var_name : str
       Name of the variable
    selection : dict[Any] -> Any
        Coordinates of the variable
    position : str
        Whether to position the coordinates' label "below" (default) or "beside"
        the name of the variable
    Returns
    -------
    label
        A text representation of the label
    """
    if selection:
        sel = selection_to_string(selection)
        if position == "below":
            sep = "\n"
        elif position == "beside":
            sep = " "
    else:
        sep = sel = ""
    return "{}{}{}".format(var_name, sep, sel)


def selection_to_string(selection):
    """Convert dictionary of coordinates to a string for labels.
    Parameters
    ----------
    selection : dict[Any] -> Any
    Returns
    -------
    str
        key1: value1, key2: value2, ...
    """
    return ", ".join(["{}".format(v) for _, v in selection.items()])


def xarray_sel_iter(
    data, var_names=None, combined=False, skip_dims=None, reverse_selections=False
):
    """Convert xarray data to an iterator over variable names and selections.
    Iterates over each var_name and all of its coordinates, returning the variable
    names and selections that allow properly obtain the data from ``data`` as desired.
    Parameters
    ----------
    data : xarray.Dataset
        Posterior data in an xarray
    var_names : iterator of strings (optional)
        Should be a subset of data.data_vars. Defaults to all of them.
    combined : bool
        Whether to combine chains or leave them separate
    skip_dims : set
        dimensions to not iterate over
    reverse_selections : bool
        Whether to reverse selections before iterating.
    Returns
    -------
    Iterator of (var_name: str, selection: dict(str, any))
        The string is the variable name, the dictionary are coordinate names to values,.
        To get the values of the variable at these coordinates, do
        ``data[var_name].sel(**selection)``.
    """
    if skip_dims is None:
        skip_dims = set()

    if combined:
        skip_dims = skip_dims.union({"chain", "draw"})
    else:
        skip_dims.add("draw")

    if var_names is None:
        if isinstance(data, xr.Dataset):
            var_names = list(data.data_vars)
        elif isinstance(data, xr.DataArray):
            var_names = [data.name]
            data = {data.name: data}

    for var_name in var_names:
        if var_name in data:
            new_dims = _dims(data, var_name, skip_dims)
            vals = [purge_duplicates(data[var_name][dim].values) for dim in new_dims]
            dims = _zip_dims(new_dims, vals)
            if reverse_selections:
                dims = reversed(dims)

            for selection in dims:
                yield var_name, selection


def xarray_var_iter(
    data, var_names=None, combined=False, skip_dims=None, reverse_selections=False
):
    """Convert xarray data to an iterator over vectors.
    Iterates over each var_name and all of its coordinates, returning the 1d
    data.
    Parameters
    ----------
    data : xarray.Dataset
        Posterior data in an xarray
    var_names : iterator of strings (optional)
        Should be a subset of data.data_vars. Defaults to all of them.
    combined : bool
        Whether to combine chains or leave them separate
    skip_dims : set
        dimensions to not iterate over
    reverse_selections : bool
        Whether to reverse selections before iterating.
    Returns
    -------
    Iterator of (str, dict(str, any), np.array)
        The string is the variable name, the dictionary are coordinate names to values,
        and the array are the values of the variable at those coordinates.
    """
    data_to_sel = data
    if var_names is None and isinstance(data, xr.DataArray):
        data_to_sel = {data.name: data}

    for var_name, selection in xarray_sel_iter(
        data,
        var_names=var_names,
        combined=combined,
        skip_dims=skip_dims,
        reverse_selections=reverse_selections,
    ):
        yield var_name, selection, data_to_sel[var_name].sel(**selection).values


def xarray_to_ndarray(data, *, var_names=None, combined=True):
    """Take xarray data and unpacks into variables and data into list and numpy array respectively.
    Assumes that chain and draw are in coordinates
    Parameters
    ----------
    data: xarray.DataSet
        Data in an xarray from an InferenceData object. Examples include posterior or sample_stats
    var_names: iter
        Should be a subset of data.data_vars not including chain and draws. Defaults to all of them
    combined: bool
        Whether to combine chain into one array
    Returns
    -------
    var_names: list
        List of variable names
    data: np.array
        Data values
    """
    data_to_sel = data
    if var_names is None and isinstance(data, xr.DataArray):
        data_to_sel = {data.name: data}

    iterator1, iterator2 = itertools.tee(
        xarray_sel_iter(data, var_names=var_names, combined=combined)
    )
    vars_and_sel = list(iterator1)
    unpacked_var_names = [
        make_label(var_name, selection) for var_name, selection in vars_and_sel
    ]

    # Merge chains and variables, check dtype to be compatible with divergences data
    data0 = data_to_sel[vars_and_sel[0][0]].sel(**vars_and_sel[0][1])
    unpacked_data = np.empty((len(unpacked_var_names), data0.size), dtype=data0.dtype)
    for idx, (var_name, selection) in enumerate(iterator2):
        unpacked_data[idx] = data_to_sel[var_name].sel(**selection).values.flatten()

    return unpacked_var_names, unpacked_data
