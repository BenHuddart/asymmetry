#!/bin/bash
# Quick script to open the documentation in the default browser

cd "$(dirname "$0")"

if [ ! -d "_build/html" ]; then
    echo "Building documentation..."
    make html
fi

if command -v open &> /dev/null; then
    # macOS
    open _build/html/index.html
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open _build/html/index.html
elif command -v start &> /dev/null; then
    # Windows
    start _build/html/index.html
else
    echo "Documentation built. Open docs/_build/html/index.html in your browser."
fi
