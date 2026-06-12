User Functions
==============

A fit function Asymmetry doesn't ship is one Python file away. You write a
vectorised function, register it with
:func:`asymmetry.register_component`, and drop the file in
``~/.asymmetry/user_functions/`` — at the next start it appears in the
fit-function builder (under *User*, badged ``· user``), fits like any
built-in component, exports through GLE plot labels, and survives project
save/load. No rebuild, no packaging, no changes to Asymmetry itself. This
replaces both of WiMDA's plugin DLL mechanisms (``musrfunctions.dll``
picker entries and ``*fit.dll`` model libraries) with plain Python.

Zero to fitted
--------------

The worked example re-implements the shipped ``Keren`` component — the
analytic dynamic Gaussian relaxation in a longitudinal field [1]_ — and is
verified in the test suite to match it bit for bit. Replace the body and
metadata with your own physics:

.. literalinclude:: examples/keren_user_function.py
   :language: python

Three steps:

1. Save the file in ``~/.asymmetry/user_functions/`` (create the folder if
   it doesn't exist; any name ending in ``.py`` works, names starting with
   ``_`` are skipped).
2. Restart Asymmetry. The log panel reports
   ``N user function(s) registered``; **Setup → User Functions…** shows the
   full load report at any time.
3. Open the fit-function builder — ``KerenUser`` is in the *User* submenu.
   Build ``KerenUser + Constant``, press Fit, and read off Δ and ν as
   usual, e.g. Δ = 0.51(2) µs⁻¹ at 20 G.

The function contract
---------------------

``register_component(name, function, param_names, *, domain, description,
formula_template, ...)`` validates everything **at load time** — a broken
file can never crash a fit (or the application) later:

* ``function(x, **params)`` must be vectorised: ``x`` is an ndarray (time
  in µs for ``domain="time"``, frequency in MHz for ``"frequency"``), one
  keyword argument per entry of ``param_names``, ndarray of the same shape
  back.
* The output must be finite on a probe grid at the default parameter
  values (NaN/Inf is rejected with a message naming the file).
* The name must be a bare identifier usable in builder expressions, and
  must be unique across **all** of Asymmetry's function registries — that
  is why the example is ``KerenUser``, not ``Keren``.
* ``domain`` is required; it places the component in the matching picker
  and plots. Optional metadata (``latex_equation``, ``applicability``,
  ``references``, ``category``, ``fixed_params``, ``param_defaults``)
  gives the component the same info-dialog documentation as a built-in.

Parameter-trend components — functions of temperature or field for the
parameter-trending builder, including the ⊕ quadrature grammar — register
through the sibling :func:`asymmetry.register_parameter_component`, whose
``scopes`` argument (``"temperature"``, ``"field"``, ``"common"``)
controls where the component is offered.

Failures, scripts, and sharing
------------------------------

Anything that goes wrong — a syntax error, a failed validation, a name
collision — is confined to that file: the rest of your plugins and the
application load normally, the log panel carries one line per failure, and
**Setup → User Functions…** shows the full error text. Fix the file and
restart (files are imported once at startup; there is no hot reload).

In analysis scripts, load your plugin directory explicitly::

    import asymmetry
    asymmetry.load_user_functions()          # ~/.asymmetry/user_functions
    # or: asymmetry.register_component(...)  # register directly, no file

To share functions as an installable package, expose a callable that
performs the registrations under the ``asymmetry.user_functions``
entry-point group; installed packages load automatically at startup.

A project that references a user function which is not installed (a
colleague's ``.asymp``, or your own after removing a plugin) opens with
the model intact: the missing component is shown by name, plots as zero,
and fitting is blocked with a message saying which function to restore.
Saving the project preserves the original model unchanged.

User functions are ordinary Python executed with full privileges — the
same trust model as WiMDA's plugin DLLs. Only install files you trust.

.. [1] A. Keren, Phys. Rev. B **50**, 10039 (1994).
