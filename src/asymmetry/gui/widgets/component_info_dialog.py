"""Reusable component-info dialog for fit-function builders."""

from __future__ import annotations

import html
from functools import lru_cache

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from asymmetry.core.fitting.component_docs import get_component_applicability
from asymmetry.core.fitting.composite import ComponentDefinition
from asymmetry.core.fitting.parameter_models import ParameterModelComponentDefinition
from asymmetry.core.fitting.parameters import ParamInfo, get_param_info
from asymmetry.gui.utils.latex_renderer import render_latex_to_html_image

ComponentDocDefinition = ComponentDefinition | ParameterModelComponentDefinition


def _param_info(component: ComponentDocDefinition, pname: str) -> ParamInfo:
    return component.param_info.get(pname, get_param_info(pname))


def _equation_html(component: ComponentDocDefinition, *, render_latex_images: bool) -> str:
    equation = component.latex_equation.strip() if component.latex_equation else ""
    if not equation:
        return f"<code>{html.escape(component.formula_template)}</code>"

    if render_latex_images:
        image_html = render_latex_to_html_image(equation, font_size=16, dpi=170)
        if image_html is not None:
            return image_html
    return f"<code>{html.escape(equation)}</code>"


def _availability_text(component: ComponentDocDefinition) -> str:
    scopes = getattr(component, "scopes", ())
    if not scopes:
        return "Time-domain fit builder"

    mapping = {
        "common": "Parameter-trending builder (all x types)",
        "field": "Parameter-trending builder for field-dependent fits",
        "temperature": "Parameter-trending builder for temperature-dependent fits",
    }
    labels = [mapping.get(scope, scope) for scope in scopes]
    return "; ".join(labels)


def _symbol_html(info: ParamInfo, *, render_latex_images: bool) -> str:
    if render_latex_images:
        image_html = render_latex_to_html_image(info.latex, font_size=13, dpi=170)
        if image_html is not None:
            return image_html
    return html.escape(info.unicode)


@lru_cache(maxsize=512)
def _build_component_info_html_cached(
    cache_key: tuple[object, ...],
    *,
    render_latex_images: bool,
) -> str:
    (
        name,
        description,
        formula_template,
        latex_equation,
        availability,
        applicability,
        row_payload,
    ) = cache_key

    rows = ""
    for symbol_latex, symbol_unicode, pname, unit, meaning in row_payload:
        symbol_info = ParamInfo(
            name=pname,
            plain=pname,
            unicode=symbol_unicode,
            latex=symbol_latex,
            gle=pname,
        )
        symbol_cell = _symbol_html(symbol_info, render_latex_images=render_latex_images)
        rows += (
            "<tr>"
            f"<td>{symbol_cell}</td>"
            f"<td><code>{html.escape(pname)}</code></td>"
            f"<td>{html.escape(unit)}</td>"
            f"<td>{html.escape(meaning)}</td>"
            "</tr>"
        )

    table = (
        "<table border='1' cellspacing='0' cellpadding='5' style='border-collapse: collapse; width: 100%;'>"
        "<tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Description</th></tr>"
        f"{rows}"
        "</table>"
    )

    proxy = type("_DocProxy", (), {
        "latex_equation": latex_equation,
        "formula_template": formula_template,
    })

    return (
        f"<h2>{html.escape(name)}</h2>"
        "<h3>Model Expression</h3>"
        f"{_equation_html(proxy, render_latex_images=render_latex_images)}"
        "<h3>Parameters</h3>"
        f"{table}"
        "<h3>Applicability</h3>"
        f"<p>{html.escape(applicability)}</p>"
        f"<p style='margin-top: 1.0em;'><i>{html.escape(availability)}</i></p>"
    )


def build_component_info_html(
    component: ComponentDocDefinition,
    *,
    render_latex_images: bool = True,
) -> str:
    """Build rich HTML documentation for a model component."""
    row_payload: list[tuple[str, str, str, str, str]] = []
    for pname in component.param_names:
        info = _param_info(component, pname)
        unit = info.unit if info.unit else "-"
        meaning = info.description or "No parameter description available."
        row_payload.append((info.latex, info.unicode, pname, unit, meaning))

    cache_key: tuple[object, ...] = (
        component.name,
        component.description,
        component.formula_template,
        component.latex_equation,
        _availability_text(component),
        get_component_applicability(component.name),
        tuple(row_payload),
    )
    return _build_component_info_html_cached(cache_key, render_latex_images=render_latex_images)


def show_component_info_dialog(parent: QWidget, component: ComponentDocDefinition) -> QDialog:
    """Open a non-modal info dialog for a selected component."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Component Info: {component.name}")
    dialog.resize(820, 600)

    layout = QVBoxLayout(dialog)
    browser = QTextBrowser(dialog)
    browser.setOpenExternalLinks(False)
    # Paint quickly with plain math text first; rendered equation image is filled in next event cycle.
    browser.setHtml(build_component_info_html(component, render_latex_images=False))
    layout.addWidget(browser)

    close_btn = QPushButton("Close", dialog)
    close_btn.clicked.connect(dialog.close)
    layout.addWidget(close_btn)

    dialog.setModal(False)
    dialog.show()

    QTimer.singleShot(
        0,
        lambda: browser.setHtml(build_component_info_html(component, render_latex_images=True)),
    )

    return dialog
