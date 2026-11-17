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

      pip install -e ".[dev]"

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b feature-name

Code style
----------

We use:

* **ruff** for linting and formatting
* **Type hints** for function signatures
* **Docstrings** in NumPy or Google style

Run checks:

.. code-block:: bash

   ruff check src/
   ruff format src/

Testing
-------

Write tests for new features using pytest:

.. code-block:: bash

   pytest tests/

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

Source mode needs no private corpus, capture, or inventory network access.
Registered missing screenshots appear as **Optional generated screenshot** notices.
The gate rejects malformed markup, broken or ambiguous authored local/API links,
unknown images, missing static/download assets, and other Sphinx warnings.
Named external objects may remain unlinked, and automatic annotation hyperlinks
are not generated; parameter and return descriptions remain readable.

For publishing, prepare ordinary GUI captures and any permitted labelled corpus
placeholders, then run ``python3 tools/harness.py docs --mode publish``. This
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
For two PRs, the first references the issue and the final PR closes it.

**Instrument review** captures the instrument, version reviewed, sections checked,
findings, and supporting evidence. Choose **Do not name me** to withhold consent
for naming you in the published review record. The issue and GitHub account remain
public; a maintainer can file on a reviewer's behalf. Consent can be changed by
commenting.

For issues filed through the CLI or API, use the form's field labels as
third-level headings, in order.

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
