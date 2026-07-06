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

Report bugs and feature requests on the GitHub issue tracker.

Please include:

* Description of the issue
* Steps to reproduce
* Expected vs actual behaviour
* System information (OS, Python version)

Code of Conduct
---------------

Be respectful and inclusive. We follow the Python Community Code of Conduct.
