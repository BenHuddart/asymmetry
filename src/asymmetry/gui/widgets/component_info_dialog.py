"""Reusable component-info dialog for fit-function builders."""

from __future__ import annotations

import html
from functools import lru_cache

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QDialog, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from asymmetry.core.fitting.component_docs import get_component_applicability
from asymmetry.core.fitting.composite import ComponentDefinition
from asymmetry.core.fitting.parameter_models import ParameterModelComponentDefinition
from asymmetry.core.fitting.parameters import ParamInfo, get_param_info
from asymmetry.gui.utils.latex_renderer import render_latex_to_html_image

ComponentDocDefinition = ComponentDefinition | ParameterModelComponentDefinition


_SC_KERNEL_LATEX = (
    r"\rho_s(T)=1+2\left\langle\int_{\Delta(T,\mathbf{k})}^{\infty}"
    r"\frac{\partial f}{\partial E}\frac{E\,dE}{\sqrt{E^2-\Delta^2(T,\mathbf{k})}}"
    r"\right\rangle_{FS}"
)

_SC_KERNEL_TEXT = (
    "This kernel defines normalized superfluid density. In TF-muSR, the superconducting "
    "Gaussian relaxation rate tracks the second moment of the vortex-lattice field distribution, "
    "so rho_s(T) controls the temperature dependence of sigma(T)."
)

_SC_GAP_MODEL_LATEX: dict[str, str] = {
    "SC_SWave": r"g(\phi)=1",
    "SC_DWave": r"g(\phi)=\cos(2\phi)",
    "SC_AnisotropicS_Cos4": r"g(\phi)=1+a\cos(4\phi)",
    "SC_NonmonotonicD": r"g(\phi)=\beta\cos(2\phi)+(1-\beta)\cos(6\phi)",
    "SC_SPlusG": r"g(\theta,\phi)=\frac{1-\sin^4\theta\cos(4\phi)}{2}",
    "SC_PWaveAxial": r"g(\phi)=\cos(\phi)",
    "SC_ExtendedS": r"g(\phi)=\cos(2\phi)\ \text{or}\ |\cos(2\phi)|",
    "SC_AlphaModel": r"g(\phi)=1,\quad \Delta_0/(k_B T_c)=1.764\,\alpha_{sc}",
    "SC_TwoGap_SS": r"\rho_s(T)=w\rho_1(T)+(1-w)\rho_2(T),\quad g_1=g_2=1",
    "SC_TwoGap_SD": r"\rho_s(T)=w\rho_s^{(s)}(T)+(1-w)\rho_s^{(d)}(T),\quad g_s=1,\ g_d=\cos(2\phi)",
    "SC_SWave_Q": r"g(\phi)=1",
    "SC_DWave_Q": r"g(\phi)=\cos(2\phi)",
    "SC_SPlusG_Q": r"g(\theta,\phi)=\frac{1-\sin^4\theta\cos(4\phi)}{2}",
}

_SC_SIGMA_MIXING_LATEX: dict[str, str] = {
    "SC_SWave": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_DWave": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_AnisotropicS_Cos4": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_NonmonotonicD": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_SPlusG": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_PWaveAxial": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_ExtendedS": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_AlphaModel": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_TwoGap_SS": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_TwoGap_SD": r"\sigma(T)=\sigma_0\rho_s(T)+\sigma_{bg}",
    "SC_SWave_Q": r"\sigma(T)=\sqrt{(\sigma_{sc}\rho_s(T))^2+\sigma_{nm}^2}",
    "SC_DWave_Q": r"\sigma(T)=\sqrt{(\sigma_{sc}\rho_s(T))^2+\sigma_{nm}^2}",
    "SC_SPlusG_Q": r"\sigma(T)=\sqrt{(\sigma_{sc}\rho_s(T))^2+\sigma_{nm}^2}",
}

_SC_GAP_MODEL_TEXT: dict[str, str] = {
    "SC_SWave": (
        "Fully gapped isotropic baseline. Use when low-temperature behavior is activated and no nodal "
        "signatures are required."
    ),
    "SC_DWave": (
        "Line-node d_{x^2-y^2} model. Use when low-temperature data are inconsistent with activated behavior "
        "and nodal quasiparticles are expected."
    ),
    "SC_AnisotropicS_Cos4": (
        "Fourfold anisotropic s-wave model. It is nodeless for |a| < 1 and can develop accidental nodes for larger |a|. "
        "The optional shape_factor_a parameter controls the generalized weak-coupling reduced-gap law: if "
        "shape_factor_a is left at 0, the model falls back to the Carrington-Manzano interpolation; if the user sets "
        "a positive fixed value or allows it to vary, that positive value is used instead."
    ),
    "SC_NonmonotonicD": (
        "Nonmonotonic d-wave extension that mixes harmonics to capture curvature not reproduced by a simple cos(2phi) form."
    ),
    "SC_SPlusG": (
        "Anisotropic singlet s+g form used when pure isotropic s-wave and pure d-wave are both too restrictive."
    ),
    "SC_PWaveAxial": (
        "Axial p-wave example for unconventional/odd-parity scenarios motivated by symmetry or complementary probes. "
        "The optional shape_factor_a parameter controls the reduced-gap amplitude law: if shape_factor_a is left at 0, "
        "the model uses the Carrington-Manzano interpolation; a positive fixed or fitted value switches to the "
        "generalized weak-coupling form."
    ),
    "SC_ExtendedS": (
        "Extended-s phenomenology useful for data that sit between simple isotropic s-wave and nodal alternatives. "
        "In this implementation the cos(2phi) basis uses the generalized weak-coupling reduced-gap form with the "
        "d-wave-like shape factor a = 4/3, rather than the Carrington-Manzano interpolation."
    ),
    "SC_AlphaModel": (
        "Single-gap BCS shape with adjustable coupling strength. The alpha parameter rescales the weak-coupling gap ratio."
    ),
    "SC_TwoGap_SS": (
        "Two-gap weighted model (MgB2-style) where two isotropic channels contribute with weight w and 1-w."
    ),
    "SC_TwoGap_SD": (
        "Mixed-symmetry weighted model combining an isotropic channel and a d-wave channel."
    ),
    "SC_SWave_Q": (
        "Uses the same isotropic gap structure as SC_SWave with quadrature linewidth mixing."
    ),
    "SC_DWave_Q": (
        "Uses the same d-wave gap structure as SC_DWave with quadrature linewidth mixing."
    ),
    "SC_SPlusG_Q": (
        "Uses the same s+g gap structure as SC_SPlusG with quadrature linewidth mixing."
    ),
}

_SC_SIGMA_MIXING_TEXT: dict[str, str] = {
    "SC_SWave": "Additive convention with superconducting scale sigma_0 and background sigma_bg.",
    "SC_DWave": "Additive convention with superconducting scale sigma_0 and background sigma_bg.",
    "SC_AnisotropicS_Cos4": "Additive convention with superconducting scale sigma_0 and background sigma_bg.",
    "SC_NonmonotonicD": "Additive convention with superconducting scale sigma_0 and background sigma_bg.",
    "SC_SPlusG": "Additive convention with superconducting scale sigma_0 and background sigma_bg.",
    "SC_PWaveAxial": "Additive convention with superconducting scale sigma_0 and background sigma_bg.",
    "SC_ExtendedS": "Additive convention with superconducting scale sigma_0 and background sigma_bg.",
    "SC_AlphaModel": "Additive convention with superconducting scale sigma_0 and background sigma_bg.",
    "SC_TwoGap_SS": "Additive convention after computing weighted two-gap superfluid density.",
    "SC_TwoGap_SD": "Additive convention after computing weighted mixed-symmetry superfluid density.",
    "SC_SWave_Q": (
        "Quadrature convention for independent Gaussian linewidth channels. Use when superconducting and "
        "non-superconducting contributions are better combined at the variance level."
    ),
    "SC_DWave_Q": (
        "Quadrature convention for independent Gaussian linewidth channels with nodal d-wave temperature dependence."
    ),
    "SC_SPlusG_Q": (
        "Quadrature convention for independent Gaussian linewidth channels with s+g anisotropic temperature dependence."
    ),
}


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


def _latex_block_html(latex: str, *, render_latex_images: bool, font_size: int = 15) -> str:
    if not latex.strip():
        return ""
    if render_latex_images:
        image_html = render_latex_to_html_image(latex, font_size=font_size, dpi=170)
        if image_html is not None:
            return image_html
    return f"<code>{html.escape(latex)}</code>"


def _physics_payload(component_name: str) -> tuple[tuple[str, str, str], ...]:
    if not component_name.startswith("SC_"):
        return tuple()

    payload: list[tuple[str, str, str]] = [("Superfluid-Density Kernel", _SC_KERNEL_LATEX, _SC_KERNEL_TEXT)]

    gap_latex = _SC_GAP_MODEL_LATEX.get(component_name)
    if gap_latex:
        payload.append(
            (
                "Gap Function / Model Form",
                gap_latex,
                _SC_GAP_MODEL_TEXT.get(component_name, "Model-specific gap symmetry used in the superfluid-density kernel."),
            )
        )

    sigma_latex = _SC_SIGMA_MIXING_LATEX.get(component_name)
    if sigma_latex:
        payload.append(
            (
                "Measured Linewidth Convention",
                sigma_latex,
                _SC_SIGMA_MIXING_TEXT.get(component_name, "Conversion from rho_s(T) to measured linewidth convention."),
            )
        )

    return tuple(payload)


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
        physics_payload,
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

    physics_html = ""
    for heading, latex, explainer in physics_payload:
        physics_html += (
            f"<h3>{html.escape(heading)}</h3>"
            f"{_latex_block_html(latex, render_latex_images=render_latex_images)}"
            f"<p>{html.escape(explainer)}</p>"
        )

    return (
        f"<h2>{html.escape(name)}</h2>"
        "<h3>Model Expression</h3>"
        f"{_equation_html(proxy, render_latex_images=render_latex_images)}"
        "<h3>Parameters</h3>"
        f"{table}"
        f"{physics_html}"
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
        _physics_payload(component.name),
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
    browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
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
