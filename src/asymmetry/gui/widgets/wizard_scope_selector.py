"""Fit-wizard scope selector — preset picker + per-family component tree.

This widget lets the user pick a *scope preset* (or "Auto (from run metadata)")
for the fit-wizard's screening pass, then optionally include/exclude individual
component families or components on top of that preset. It renders exactly what
an injected *resolver* returns — it holds **no** physics logic;
:func:`resolver_payload` adapts the core ``ScopeView`` to the plain-dict
contract below.

Resolver contract
-----------------
The widget is driven by a single injected callable::

    resolver(preset_id: str, overrides: dict) -> dict

where ``preset_id`` is one of the :data:`PRESET_CHOICES` ids (e.g. ``"auto"``)
and ``overrides`` is::

    {"include": [component_name, ...], "exclude": [component_name, ...]}

— the user's deltas relative to the preset's own baseline selection. The
resolver returns a plain ``dict``::

    {
      "note": str,               # metadata read-back shown under the combo,
                                 #   e.g. "TF, 20 G, 5 K — Auto selected precession families"
      "families": [
        {
          "key": str,            # stable family id
          "title": str,          # human family title, shown on the parent row
          "components": [
            {
              "name": str,       # component name, shown on the child row (col 0)
              "included": bool,  # baseline check state for this component
              "reason": str,     # non-empty ONLY for excluded components; tooltip text
              "cost": str,       # cost hint, shown in col 1 (e.g. "1 fit", "cheap")
            },
            ...
          ],
        },
        ...
      ],
      "estimate": int,           # number of components that will be screened
    }

``reason`` is non-empty only for *excluded* components and becomes that row's
tooltip. The widget never invents a reason, cost, or inclusion decision.

Resolver call discipline
-------------------------
The widget resolves twice per refresh:

* ``resolver(preset, {})`` — the **baseline** resolution. Its ``families`` build
  the tree and its per-component ``included`` flags are the reference against
  which user check-state edits are diffed into overrides. This resolution's
  ``note`` labels the metadata line.
* ``resolver(preset, overrides)`` — used **only** to read ``estimate`` (and,
  when overrides exist, ``note``) for the labels. Its ``families`` are ignored,
  so the widget behaves identically whether or not a resolver reflects overrides
  back in its family list.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.wizard_scope import ScopeView, WizardScope
from asymmetry.core.workflow.screen import SCOPE_PRESETS
from asymmetry.gui.styles import tokens

#: A resolver as documented in this module's docstring.
Resolver = Callable[[str, dict], dict]

#: (id, label) for every scope preset offered in the combo. "auto" is default.
PRESET_CHOICES: tuple[tuple[str, str], ...] = (
    ("auto", "Auto (from run metadata)"),
    ("zf-static-magnetism", "ZF static magnetism"),
    ("tf-knight-precession", "TF Knight shift / precession"),
    ("tf-superconductor", "TF superconductor"),
    ("lf-dynamics", "LF dynamics"),
    ("fluoride-fmuf", "Fluoride (F-μ-F)"),
    ("muonium-radical", "Muonium / radical"),
    ("all", "All components"),
)

#: Preset id for each physics set a preset offers; the first preset in combo order wins.
_PRESET_FOR_PHYSICS = {SCOPE_PRESETS[pid]: pid for pid, _ in reversed(PRESET_CHOICES)}

#: Transient combo entry shown only while user overrides exist.
_CUSTOM_LABEL = "Custom"

#: Role storing a leaf row's component name (parents leave it unset).
_COMPONENT_NAME_ROLE = Qt.ItemDataRole.UserRole


# Phase 4 replaces this with ModelFamilyPicker.
def scope_for_preset(preset: str, include: set[str], exclude: set[str]) -> WizardScope:
    """The scope a preset id stands for: the CLI's physics shortcut plus overrides."""
    return WizardScope(
        physics=SCOPE_PRESETS[preset],
        include_components=frozenset(include),
        exclude_components=frozenset(exclude),
    )


def build_scope_payload(preset: str, include: set[str], exclude: set[str]) -> dict:
    """Return the serialised :class:`WizardScope` payload for the given state."""
    return scope_for_preset(preset, include, exclude).to_payload()


# Phase 4 replaces this with ModelFamilyPicker.
def parse_scope_payload(scope: dict | None) -> tuple[str, set[str], set[str]]:
    """Parse a scope payload into ``(preset_id, include, exclude)``.

    ``None`` resolves to ``"auto"`` with no overrides; a payload's physics maps
    back to the first preset offering it.
    """
    if scope is None:
        return "auto", set(), set()
    parsed = WizardScope.from_payload(scope)
    return (
        _PRESET_FOR_PHYSICS[parsed.physics],
        set(parsed.include_components),
        set(parsed.exclude_components),
    )


# Phase 4 replaces this with ModelFamilyPicker.
def resolver_payload(view: ScopeView) -> dict:
    """Adapt a core :class:`ScopeView` to the resolver dict contract above."""
    return {
        "note": "; ".join(view.notes),
        "families": [
            {
                "key": family.title,
                "title": family.title,
                "components": [
                    {
                        "name": component.name,
                        "included": component.included,
                        "reason": component.reason,
                        "cost": "slow" if component.slow else "",
                    }
                    for component in family.components
                ],
            }
            for family in view.families
        ],
        "estimate": view.included_count,
    }


class WizardScopeSelector(QWidget):
    """Preset picker plus per-family component include/exclude tree.

    Emits :attr:`scope_changed` (serialised payload dict) after any user edit,
    and :attr:`validity_changed(is_valid())` alongside it on tree edits (its
    bool carries the current validity — it is not an edge/boundary detector).
    Programmatic changes (``refresh_from_context``, ``set_scope``) never emit.
    """

    scope_changed = Signal(object)  # serialised scope payload dict
    validity_changed = Signal(bool)  # False when zero components are included

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._resolver: Resolver | None = None
        self._preset_id: str = "auto"
        self._overrides: dict[str, set[str]] = {"include": set(), "exclude": set()}
        # Baseline included-flags from the last resolver(preset, {}) call, keyed
        # by component name — the reference for diffing user check-state edits.
        self._baseline_included: dict[str, bool] = {}
        self._custom_present: bool = False
        # Last payload emitted via scope_changed, to coalesce the duplicate
        # itemChanged events a single check-state edit produces (Qt fires one for
        # the leaf and one for its auto-tristate parent).
        self._last_emitted_scope: dict | None = None

        # ── Preset row ────────────────────────────────────────────────────────
        self._preset_combo = QComboBox(self)
        for pid, label in PRESET_CHOICES:
            self._preset_combo.addItem(label, userData=pid)
        self._preset_combo.setCurrentIndex(0)  # "auto"
        # Connect only AFTER the initial population/default so the -1→0 index
        # move above does not reach the handler (which would clear overrides).
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.addWidget(QLabel("Preset:", self))
        preset_row.addWidget(self._preset_combo, 1)

        # ── Metadata read-back ────────────────────────────────────────────────
        self._metadata_label = QLabel("", self)
        self._metadata_label.setWordWrap(True)
        self._metadata_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        # ── Family / component tree ───────────────────────────────────────────
        self._family_tree = QTreeWidget(self)
        self._family_tree.setColumnCount(2)
        self._family_tree.setHeaderLabels(["Family / Component", "Cost"])
        self._family_tree.setRootIsDecorated(True)
        # The name column takes all spare width (long component names must stay
        # readable); the cost column hugs its short labels.
        header = self._family_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._family_tree.itemChanged.connect(self._on_item_changed)

        # ── Estimate / validity line ──────────────────────────────────────────
        self._estimate_label = QLabel("", self)
        self._estimate_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(preset_row)
        layout.addWidget(self._metadata_label)
        layout.addWidget(self._family_tree, 1)
        layout.addWidget(self._estimate_label)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_resolver(self, resolver: Resolver | None) -> None:
        """Store *resolver* for subsequent refreshes. Does not auto-refresh."""
        self._resolver = resolver

    def refresh_from_context(self) -> None:
        """Re-resolve and repopulate the tree/labels. Emits nothing.

        Call after :meth:`set_resolver` or whenever the underlying dataset /
        run metadata changes.
        """
        self._repopulate()

    def current_scope(self) -> dict:
        """Return the current scope as a serialised payload dict."""
        return build_scope_payload(
            self._preset_id,
            self._overrides["include"],
            self._overrides["exclude"],
        )

    def set_scope(self, scope: dict | None) -> None:
        """Restore state from a cached payload. Emits no signals.

        ``None`` resets to ``"auto"`` with no overrides. If a resolver is set,
        the tree/labels are refreshed to match.
        """
        preset, include, exclude = parse_scope_payload(scope)
        self._preset_id = preset
        self._overrides = {"include": include, "exclude": exclude}
        # Forget the last emitted payload so the next genuine user edit is not
        # coalesced against a stale value from before this restore.
        self._last_emitted_scope = None
        if self._resolver is not None:
            self._repopulate()

    def is_valid(self) -> bool:
        """Return whether at least one component is currently included."""
        return self._included_leaf_count() >= 1

    # ── Population (never emits) ───────────────────────────────────────────────

    def _repopulate(self) -> None:
        """Re-resolve and rebuild the combo/tree/labels under signal blockers.

        The baseline resolution (``resolver(preset, {})``) supplies the tree
        rows and the reference inclusion flags; the override resolution supplies
        the estimate and (when overrides exist) the metadata note.
        """
        if self._resolver is None:
            return

        overrides_payload = {
            "include": sorted(self._overrides["include"]),
            "exclude": sorted(self._overrides["exclude"]),
        }
        baseline = self._resolver(self._preset_id, {"include": [], "exclude": []})
        resolved = self._resolver(self._preset_id, overrides_payload)

        self._baseline_included = {}

        # Combo: reflect the real preset; add/select transient "Custom" only
        # while overrides exist. All combo mutation is blocked so the preset
        # handler (which clears overrides) never fires programmatically.
        has_overrides = bool(self._overrides["include"] or self._overrides["exclude"])
        with QSignalBlocker(self._preset_combo):
            self._sync_custom_item(has_overrides)
            self._select_preset_or_custom(has_overrides)

        note = resolved.get("note") if has_overrides else baseline.get("note")
        self._metadata_label.setText(str(note or ""))

        with QSignalBlocker(self._family_tree):
            self._family_tree.clear()
            for family in baseline.get("families", []) or []:
                parent = QTreeWidgetItem(self._family_tree)
                parent.setText(0, str(family.get("title", "")))
                parent.setText(1, "")
                parent.setFlags(
                    parent.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsAutoTristate
                )
                for component in family.get("components", []) or []:
                    name = str(component.get("name", ""))
                    included_baseline = bool(component.get("included", False))
                    self._baseline_included[name] = included_baseline
                    # Effective check state = baseline flag adjusted by overrides.
                    checked = included_baseline
                    if name in self._overrides["exclude"]:
                        checked = False
                    elif name in self._overrides["include"]:
                        checked = True

                    child = QTreeWidgetItem(parent)
                    child.setText(0, name)
                    child.setText(1, str(component.get("cost", "")))
                    child.setData(0, _COMPONENT_NAME_ROLE, name)
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(
                        0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                    )
                    reason = str(component.get("reason", "") or "")
                    if reason:
                        child.setToolTip(0, reason)
            self._family_tree.expandAll()

        self._update_estimate_label(resolved.get("estimate"))

    # ── User-edit handlers (emit) ─────────────────────────────────────────────

    def _on_preset_selected(self, index: int) -> None:
        """A real preset was picked: clear overrides, repopulate, emit once."""
        pid = self._preset_combo.itemData(index)
        if pid is None:
            # The transient "Custom" item carries no userData; ignore it.
            return
        self._preset_id = str(pid)
        self._overrides = {"include": set(), "exclude": set()}
        self._repopulate()
        payload = self.current_scope()
        self._last_emitted_scope = payload
        self.scope_changed.emit(payload)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """A user check-state edit: recompute overrides, update labels, emit.

        A single check-state edit fires this twice (leaf + auto-tristate parent);
        the recomputed payload is identical both times, so we coalesce on it and
        emit only on a genuine change.
        """
        self._recompute_overrides()
        payload = self.current_scope()
        if payload == self._last_emitted_scope:
            return

        has_overrides = bool(self._overrides["include"] or self._overrides["exclude"])
        with QSignalBlocker(self._preset_combo):
            self._sync_custom_item(has_overrides)
            self._select_preset_or_custom(has_overrides)

        # Re-resolve for the fresh estimate / note without touching the tree.
        if self._resolver is not None:
            overrides_payload = {
                "include": sorted(self._overrides["include"]),
                "exclude": sorted(self._overrides["exclude"]),
            }
            resolved = self._resolver(self._preset_id, overrides_payload)
            if has_overrides and resolved.get("note"):
                self._metadata_label.setText(str(resolved["note"]))
            self._update_estimate_label(resolved.get("estimate"))
        else:
            self._update_estimate_label(None)

        self._last_emitted_scope = payload
        self.scope_changed.emit(payload)
        self.validity_changed.emit(self.is_valid())

    # ── Internals ─────────────────────────────────────────────────────────────

    def _recompute_overrides(self) -> None:
        """Rebuild overrides as a full diff of every leaf vs the baseline.

        A full-tree diff (rather than an incremental toggle) is idempotent, which
        matters because a tri-state parent toggle fires an ``itemChanged`` for
        every child plus the parent. included-by-baseline but now unchecked →
        "exclude"; excluded-by-baseline but now checked → "include".
        """
        include: set[str] = set()
        exclude: set[str] = set()
        for leaf in self._iter_leaves():
            name = leaf.data(0, _COMPONENT_NAME_ROLE)
            if name is None:
                continue
            name = str(name)
            checked = leaf.checkState(0) == Qt.CheckState.Checked
            baseline = self._baseline_included.get(name, False)
            if baseline and not checked:
                exclude.add(name)
            elif not baseline and checked:
                include.add(name)
        self._overrides = {"include": include, "exclude": exclude}

    def _iter_leaves(self):
        """Yield every component (leaf) item in the tree."""
        root = self._family_tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            for j in range(parent.childCount()):
                yield parent.child(j)

    def _included_leaf_count(self) -> int:
        return sum(1 for leaf in self._iter_leaves() if leaf.checkState(0) == Qt.CheckState.Checked)

    def _sync_custom_item(self, has_overrides: bool) -> None:
        """Add or remove the transient "Custom" combo item. Combo must be blocked."""
        if has_overrides and not self._custom_present:
            self._preset_combo.addItem(_CUSTOM_LABEL, userData=None)
            self._custom_present = True
        elif not has_overrides and self._custom_present:
            idx = self._preset_combo.findData(None)
            if idx >= 0:
                self._preset_combo.removeItem(idx)
            self._custom_present = False

    def _select_preset_or_custom(self, has_overrides: bool) -> None:
        """Select "Custom" while overrides exist, else the real preset. Blocked."""
        if has_overrides:
            idx = self._preset_combo.findData(None)
        else:
            idx = self._preset_combo.findData(self._preset_id)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    def _update_estimate_label(self, estimate: object) -> None:
        """Show the screened-component count, or a red warning when nothing is included."""
        if not self.is_valid():
            self._estimate_label.setText("No components included — select at least one to screen.")
            self._estimate_label.setStyleSheet(f"color: {tokens.ERROR}; font-weight: 600;")
            return
        candidates = int(estimate) if isinstance(estimate, int) else 0
        self._estimate_label.setText(f"{candidates} candidates")
        self._estimate_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
