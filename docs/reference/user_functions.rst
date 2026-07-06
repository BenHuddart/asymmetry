User functions
==============

A fit function Asymmetry doesn't ship is one function away. Every fit
component and parameter-trend model is backed by an ordinary Python file
under ``~/.asymmetry/user_functions/`` — at the next start it appears in the
fit-function builder (under *User*, badged ``· user``), fits like any
built-in component, exports through GLE plot labels, and survives project
save/load. No rebuild, no packaging, no changes to Asymmetry itself —
extending the fitting library is plain Python.

You write that file yourself only if you want to; the usual route is to let
the GUI write it for you.

Building a function in the GUI
-------------------------------

.. image:: /_generated/screenshots/new_user_function_dialog.png
   :alt: New User Function dialog authoring a stretched-exponential oscillation
   :width: 100%

*The New User Function dialog mid-authoring:* ``StretchedOsc``, *a*
*stretched-exponential envelope on a Larmor oscillation. The parameter table*
*carries a start value per name; the preview redraws at those values, and the*
*green status line with an enabled OK button confirms every load-time check*
*has passed.*

Both function builders — the fit builder (Fit panel → **Build…**) and the
parameter-model builder (the parameter-trending window's model editor) —
have a **New user function…** button. It sits in the library footer, and a
second copy appears as an invitation whenever a search comes up with no
matches, so typing the name of a function that doesn't exist yet is itself a
route to creating it. Either button opens the same dialog.

The dialog asks for:

* **Name** — the identifier the function is registered and inserted under
  (e.g. ``StretchedOsc``); it must be unique across every built-in and
  user-defined component.
* **Description** — the one-line summary shown in the library and the
  component info dialog.
* **Formula** — a maths expression in ``x`` (time in μs, frequency in MHz,
  or the trend variable, depending on where you opened the dialog from) and
  your parameter names. Bare maths names (``exp``, ``sin``, ``cos``,
  ``sqrt``, ``pi``, …) and ``np.*`` both work, so ``A*exp(-lam*x)`` and
  ``A*np.exp(-lam*x)`` are equivalent.
* **Parameters** — a small table of parameter names and start values. Click
  **Detect parameters** to populate it automatically from every name in the
  formula that isn't ``x``, ``np``, or a recognised maths function; by
  convention the first parameter of a fit component is its amplitude.
* **Edit as Python (advanced)** — a toggle that replaces the single-line
  formula with a full editor, pre-filled with the exact code your formula
  would have generated. Use it for anything a one-line expression can't
  express — conditionals, piecewise definitions, a helper computation before
  the ``return``.

As you type, a preview curve redraws at the parameter start values, so a
typo or a domain mismatch shows up as a wrong-looking curve (or a validation
message) before you commit to anything. Validation is debounced — it waits
until you pause rather than running on every keystroke, because building the
preview curve executes your code. It runs exactly the checks a hand-written
plugin faces at load time — a legal, globally unique identifier, valid
parameter names (none clashing with ``x``, ``np``, or a maths function), a
non-empty description, and a finite result on a probe grid — with no second,
looser path; **OK** stays disabled until they all pass.

Press **OK** and Asymmetry writes an ordinary, readable plugin file into
``~/.asymmetry/user_functions/`` (the same folder and format described
below — hand-editable, and it reloads at every subsequent startup exactly
like a plugin you wrote yourself), registers the function immediately, and
inserts it into the model you were building. There is no extra "install"
step and no restart needed for the function you just created; **Setup →
User Functions…** lists it alongside every other loaded plugin from that
moment on.

.. note::
   Whichever route creates it, a user function is ordinary Python executed
   with full interpreter privileges. Only create or install functions you
   trust, and be as careful sharing a generated plugin file as you would
   sharing a hand-written one.

Writing the file yourself
--------------------------

The GUI dialog covers a single formula in ``x``. For anything that benefits
from being scripted, versioned, or shared as a package — a model with
helper functions, one you want under source control, or one you want to
distribute to a group — write the plugin file directly. This is the same
mechanism the GUI uses under the bonnet, so a function created either way is
indistinguishable to the rest of Asymmetry.

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
   usual, e.g. Δ = 0.51(2) μs⁻¹ at 20 G.

The function contract
---------------------

``register_component(name, function, param_names, *, domain, description,
formula_template, ...)`` validates everything **at load time** — a broken
file can never crash a fit (or the application) later:

* ``function(x, **params)`` must be vectorised: ``x`` is an ndarray (time
  in μs for ``domain="time"``, frequency in MHz for ``"frequency"``), one
  keyword argument per entry of ``param_names``, ndarray of the same shape
  back.
* The output must be finite on a probe grid at the default parameter
  values (NaN/Inf is rejected with a message naming the file).
* The name must be a bare identifier usable in builder expressions, and
  must be unique across **all** of Asymmetry's function registries — that
  is why the example is ``KerenUser``, not ``Keren``.
* Optional physics tags feed the fit wizard's scope selector:
  ``field_geometries`` (any of ``"ZF"``, ``"TF"``, ``"LF"``; default all
  three), ``physics_classes`` (e.g. ``"magnetism"``, ``"dynamics"``,
  ``"muonium"``; default ``"custom"``), and ``cost`` (``"cheap"``,
  ``"moderate"``, or ``"expensive"``; default ``"moderate"``). Untagged
  user functions match every wizard scope preset, so they are never
  hidden; tagging them narrows when the wizard auto-considers them and
  tells the tiered screener how expensive they are.
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

Removing a function is the reverse of installing one: delete its file
from the plugin folder — **Setup → User Functions…** → **Open folder…**
takes you straight there — and restart. Saved projects that referenced it
degrade exactly as above, to the named placeholder, and the function goes
live again if the file is ever restored.

User functions are ordinary Python executed with full interpreter
privileges — only install files you trust.

.. [1] A. Keren, Phys. Rev. B **50**, 10039 (1994).
