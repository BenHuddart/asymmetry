# Third-party licenses (bundled binaries)

Asymmetry itself is MIT-licensed (see [LICENSE](LICENSE)). The pre-built
desktop binaries (Windows installer, macOS DMG) bundle some third-party
runtime libraries whose licenses require their notices to be reproduced in
binary distributions. Those notices are collected here and shipped alongside
the application.

This applies only to the **packaged binaries**. When installing from PyPI/source
(`pip install asymmetry[hdf4]`), these libraries come from their own wheels
(`pyhdf` and, on Windows, a separately provided HDF4 runtime) under their own
licenses.

---

## HDF4 — `hdf.dll`, `mfhdf.dll`, `xdr.dll` (and the macOS `libhdf`/`libmfhdf` dylibs)

The HDF4 C library, used via `pyhdf` to read legacy ISIS muon NeXus v1 `.nxs`
files stored in an HDF4 container. License: BSD-3-Clause-style (The HDF Group).

```
Copyright Notice and License Terms for
Hierarchical Data Format (HDF) Software Library and Utilities
---------------------------------------------------------------------------

Hierarchical Data Format (HDF) Software Library and Utilities
Copyright 2006 by The HDF Group.

NCSA Hierarchical Data Format (HDF) Software Library and Utilities
Copyright 1988-2006 by the Board of Trustees of the University of Illinois.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted for any purpose (including commercial purposes)
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions, and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions, and the following disclaimer in the documentation
   and/or materials provided with the distribution.

3. Neither the name of The HDF Group, the name of the University, nor the
   name of any Contributor may be used to endorse or promote products derived
   from this software without specific prior written permission from The HDF
   Group, the University, or the Contributor, respectively.

DISCLAIMER:
THIS SOFTWARE IS PROVIDED BY THE HDF GROUP AND THE CONTRIBUTORS "AS IS"
WITH NO WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED.  IN NO EVENT
SHALL THE HDF GROUP OR THE CONTRIBUTORS BE LIABLE FOR ANY DAMAGES SUFFERED
BY THE USERS ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to The HDF Group, without imposing a
separate written license agreement for such Enhancements, then you hereby grant
the following license: a non-exclusive, royalty-free perpetual license to
install, use, modify, prepare derivative works, incorporate into other computer
software, distribute, and sublicense such enhancements or derivative works
thereof, in binary and source code form.

Contributors:   National Center for Supercomputing Applications (NCSA) at
the University of Illinois, Fortner Software, Unidata Program Center (netCDF),
The Independent JPEG Group (JPEG), Jean-loup Gailly and Mark Adler (gzip),
and Digital Equipment Corporation (DEC).
```

The HDF4 Windows DLLs are sourced from the conda-forge `hdf4` package (4.2.15).

---

## libjpeg-turbo — `jpeg8.dll`, `turbojpeg.dll` (Windows)

JPEG codec linked by the HDF4 library. License: a combination of the IJG
(Independent JPEG Group) license, the modified (3-clause) BSD license, and the
zlib license. See <https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/LICENSE.md>.
Sourced from the conda-forge `libjpeg-turbo` package (2.1.4).

---

## zlib

The zlib compression library (zlib license) may be linked by the HDF4 runtime.
See <https://zlib.net/zlib_license.html>.
