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

      cd docs
      make html
      open _build/html/index.html

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
