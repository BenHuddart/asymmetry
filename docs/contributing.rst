Contributing
============

We welcome contributions to Asymmetry!

Development setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/your-username/asymmetry.git
      cd asymmetry

3. Install in development mode:

   .. code-block:: bash

      pip install -c constraints.txt -e ".[all,dev]"

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b feature/my-change

Code style
----------

We use:

* **ruff** for linting and formatting
* **Type hints** for function signatures
* **Docstrings** in NumPy or Google style

Run checks:

.. code-block:: bash

   python tools/harness.py lint

Testing
-------

Place tests beside the owning layer and run them through the repository harness:

.. code-block:: bash

   python tools/harness.py test -- tests/core/test_instrument_profiles.py
   python tools/harness.py validate

Ensure all tests pass before submitting.

Documentation
-------------

Update documentation for any new features:

1. Add docstrings to new functions/classes
2. Update user guide if needed
3. Build docs locally to check:

   .. code-block:: bash

      python3 tools/harness.py docs --mode source
      open docs/_build/html/index.html

Source mode runs offline without example data or screenshot capture.
Registered missing screenshots appear as **Optional generated screenshot** notices.
The gate rejects malformed markup, broken or ambiguous authored local/API links,
unknown images, missing static/download assets, and other Sphinx warnings.
Named external objects may remain unlinked, and automatic annotation hyperlinks
are not generated; parameter and return descriptions remain readable.

For publishing, prepare ordinary GUI captures and labelled placeholders for
any missing example-data figures, then run ``python3 tools/harness.py docs --mode publish``. This
uses the same defect gate and also requires complete ordinary GUI assets.
A limited source build does not certify publishing readiness. Both modes reread
all sources, preserve build failures, and enforce a 600 KB HTML page budget.

Pull request process
---------------------

1. Ensure tests pass and code is formatted
2. Update CHANGELOG.md
3. Push to your fork
4. Open a pull request with a clear description

Reporting issues
----------------

Use the `issue forms <https://github.com/BenHuddart/Asymmetry/issues/new/choose>`_
to record work that can be implemented and checked independently.
**Implementation** captures the outcome, evidence, scope, decisions, compatibility,
acceptance criteria, validation, and documentation. List prerequisites under
**Blocked by** and start dependent work only when each is closed as completed.
Use ``Refs #n`` while acceptance criteria remain open. Only a PR satisfying the
whole issue uses ``Closes #n``.

**Instrument review** captures the instrument, version reviewed, sections checked,
findings, and supporting evidence. Choose **Do not name me** to withhold consent
for naming you in the published review record. The issue and GitHub account remain
public; a maintainer can file on a reviewer's behalf. Consent can be changed by
commenting.

For issues filed through the CLI or API, use the form's field labels as
third-level headings, in order.

.. _issue-status-and-monitoring:

Issue status and monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep an issue open with ``monitoring`` when its mitigation has shipped, further
implementation is not currently justified, and defined evidence or recurrence
will determine the next action. This status preserves the remaining uncertainty;
it does not claim that the underlying cause is understood. Use ``deferred`` for
work deliberately postponed. Keep scope and severity labels alongside the status.

Before applying ``monitoring``, record these points in the issue:

* The remaining uncertainty, shipped mitigation and supporting PR or commit.
* A named owner responsible for reviewing new evidence.
* The evidence source, including relevant CI runs, jobs and retained diagnostics.
* The escalation trigger and next review point.
* The closure criterion, including any residual uncertainty requiring an explicit
  maintainer decision.

Review monitoring issues before each release and when their escalation trigger
occurs. Inspect CI incidents promptly and preserve required findings before
diagnostic artifacts expire. Record failed attempts and successful retries;
a successful retry does not erase the failure or establish its cause.

Remove ``monitoring`` and resume active investigation when the trigger occurs
or new evidence justifies implementation. Remove ``deferred`` when postponed
work resumes. Keep these statuses distinct. An active-work search can exclude
both with ``is:issue is:open -label:monitoring -label:deferred``.

Close a monitoring issue only when its recorded criterion is met or the
maintainer explicitly accepts the remaining uncertainty. State what the evidence
establishes and what remains unknown. Inactivity alone is not a closure criterion.

Code of Conduct
---------------

Be respectful and inclusive. We follow the Python Community Code of Conduct.

Released project compatibility
------------------------------

Keep versioned project and standalone wizard converters, with synthetic fixtures
matching emitted released shapes, permanently. This includes alpha releases.
Remove runtime repair after conversion is owned by persistence and current
producers emit canonical state. Check parameter curves independently of JSON
round trips; preserve draft forms, unavailable user models, and captured execution
attribution. A storage version never proves that a historical calculation is
current.

Instrument definitions
----------------------

A profile is the reviewable authority for instrument assumptions. Add one frozen
``InstrumentProfile`` in a facility-qualified module under
``asymmetry.core.instruments``. Keep its fields in this order: identity, source,
recognition, detectors, presets, quirks, defaults, sources, and review. The
catalogue discovers bundled declarations; generated review pages need no manual
registration. Do not add loader or GUI branches for an instrument name.

Declare facts before drawing them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep physical banks/elements, acquisition channels, and display occurrences
separate. ``Channel`` supplies a stable key and the real recorded histogram
ordinal; a physical association may be empty or multiple. Facility evidence has
precedence during recognition. Aliases match metadata names; filename tokens are
a separate source of evidence. Unknown identity or geometry must remain unknown.

Each ``Acquisition`` declares its compatible channel map and ``View`` records.
Use ``Composite`` to combine flat arrangements: ``Directional`` holds ordered
``BankRow`` objects, while ``Circular``, ``Grid``, and ``Explicit`` cover other
placements. Shape and arrangement are independent. Every selectable channel has
one editable occurrence across its views/auxiliary strip; repeats are read-only
context. Put an unlocated channel in ``auxiliary`` rather than invent a position.

``BeamDirection`` and ``ViewOrientation`` define a common arrangement frame and
screen basis explicitly. Placements remain view-local. Positive toward-viewer
projection means out of the screen; a missing toward basis retains unknown axial
sense. Schematic distances are not detector dimensions, acceptance weights, or
run-specific field/spin settings. Scientific presets specify channel keys,
ordered asymmetry legs, and nominal balances independently of geometry.

An acquisition can declare alternative ``ReadoutSet`` selections using real
channel keys. ``CombinedReadout`` records known shared-event constituents;
``ConditionalContribution`` records an unresolved contributor and its possible
destinations. These are not exact count equations or a covariance model. The
core checks the effective requested channels, including their own references
and original periods; do not reproduce that eligibility rule in the front end.
A set transition must not silently remap saved groups or fitted calibration.

Work through the GPS declaration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPS's fifteen-channel acquisition is a compact example of alternative readout
sets with shared Forward/Backward and unlocated Mobile. Its declaration derives
memberships from the acquired channel order; the following excerpt is included
directly from the bundled profile, rather than maintained as another definition:

.. literalinclude:: ../src/asymmetry/core/instruments/psi_gps.py
   :language: python
   :start-at:         readout_sets=tuple(
   :end-before:         combined_readouts=tuple(
   :dedent: 8
   :caption: GPS acquisition readout-set declaration (excerpt)

Check a declaration through the same resolver and numbering API that feed the
editor and generated review material. This executable example inspects the
bundled profile; it does not assign new hardware facts:

.. code-block:: python
   :name: instrument-authoring-example

   from asymmetry.core.instruments.psi_gps import PROFILE
   from asymmetry.core.instruments.geometry import BeamDirection, ViewOrientation
   from asymmetry.core.instruments.numbering import numbering_template
   from asymmetry.core.instruments.readouts import layout_for_readout_set
   from asymmetry.core.instruments.resolve import layout_from_profile

   acquisition = next(c for c in PROFILE.detectors.acquisitions
                      if c.key == "root_combined_split")
   assert acquisition.beam == BeamDirection((1, 0, 0))
   assert acquisition.views[0].orientation == ViewOrientation((1, 0, 0), (0, 1, 0))
   layout = layout_from_profile(PROFILE, acquisition.key)
   individual = layout_for_readout_set(layout, "individual")
   slots = numbering_template(PROFILE, acquisition.key)
   assert len(slots) == 15
   assert len(individual.active_segments) == 11
   assert individual.banks[0].beam.screen == (1.0, 0.0)
   assert all(slot.id.startswith(f"{PROFILE.identity.key}/{acquisition.key}/")
              for slot in slots)
   facts = {item.fact for item in PROFILE.sources.evidence}
   assert set(PROFILE.fact_keys) <= facts

A numbering slot is qualified by profile, acquisition, and channel. Changing
view order or filtering a set does not renumber the stored histograms. Check the
labelled diagram, blank numbering diagram, and editable table for every acquired
configuration, and compare them with the actual Detector Layout editor.

Evidence and review
~~~~~~~~~~~~~~~~~~~

Attach ``Evidence`` to every named fact or coherent table. Cite a primary
``Reference`` and locator where known; mark unavailable evidence ``unverified``
rather than inferring it from a label or count. Published hardware evidence and
unverified exported channel associations can have different statuses. Preserve
those distinctions in readout relationships and source/default declarations.

The generated :doc:`instrument definitions <reference/instruments/index>` expose
the declarations, diagrams, numbering downloads, and evidence. An authored preset
is not an instrument-scientist review. Record the exact profile revision and
scope checked, and obtain consent before publishing the reviewer's name.
Maintainer acceptance of a GUI diagram does not certify its physical map.

Add a synthetic recognition/default-grouping test, independent source anchors for
scientific conventions, and controls for unknown or conflicting evidence. Use
existing schema, geometry, readout, and generated-doc tests for the shared
contract. Source-profile updates preserve recorded scientific choices; introducing
new defaults requires its own scientific decision and replay controls.
