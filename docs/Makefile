# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
SCREENSHOTDIR = _generated/screenshots
REPOROOT      := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
SCREENSHOTCMD  = cd $(REPOROOT) && QT_QPA_PLATFORM=offscreen python -m docs.screenshots.capture --out docs/$(SCREENSHOTDIR)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile screenshots html

# Regenerate GUI screenshots into docs/_generated/screenshots.
# These are not committed; the Sphinx build embeds them via _generated/.
screenshots:
	$(SCREENSHOTCMD)

# Ensure screenshots exist before the html build so image references resolve.
html: screenshots
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
