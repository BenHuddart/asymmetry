"""Confirm instrument identity without changing grouping or calculating a reduction."""

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
)

from asymmetry.core.instruments.catalog import PROFILES
from asymmetry.core.instruments.resolution import (
    GenericInstrumentIdentity,
    InstrumentSelection,
    RecordedInstrumentChoice,
    UnmappedAcquisition,
    UnresolvedInstrumentIdentity,
    accept_instrument_choice,
    resolve_instrument,
)
from asymmetry.core.project.profiles import GroupingProfile, profile_fingerprint_for_run
from asymmetry.gui.styles.metrics import dialog_width, row_height


# Origin and revision describe provenance; they do not create another selectable identity.
def _identity_label(selected) -> str:
    if isinstance(selected, GenericInstrumentIdentity):
        return "Generic / unidentified instrument"
    if isinstance(selected, UnresolvedInstrumentIdentity):
        return f"Unresolved: {selected.label}"
    profile = next((p for p in PROFILES if p.identity.key == selected.profile_key), None)
    if profile is None:
        return f"{selected.profile_key} — {selected.acquisition} (profile unavailable)"
    acquisition = selected.acquisition
    if isinstance(acquisition, UnmappedAcquisition):
        config_label = (
            "no declared channel map"
            if acquisition.reason == "undeclared"
            else "acquired channels incompatible"
        )
    else:
        config_label = next(
            (c.label for c in profile.detectors.acquisitions if c.key == acquisition), acquisition
        )
    return f"{profile.identity.name} ({profile.identity.facility}) — {config_label}"


class InstrumentIdentityDialog(QDialog):
    """Choose identity, or confirm one fixed proposal supplied by the layout editor."""

    def __init__(
        self, run, following_profile: GroupingProfile | None = None, parent=None, *, selection=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Instrument Identity")
        self.resize(dialog_width(76), self.sizeHint().height())
        self._run = run
        self._following = following_profile
        context = resolve_instrument(run)
        retained = context.recorded.selection if context.recorded is not None else None
        layout = QVBoxLayout(self)
        self._retained = QLabel(
            "Retained identity: " + (_identity_label(retained) if retained is not None else "None")
        )
        outcome = (
            " — explicit choice required by conflicting evidence" if context.requires_choice else ""
        )
        self._observed = QLabel(
            "Observed candidates: "
            + ("; ".join(_identity_label(item) for item in context.observed.candidates) or "None")
            + outcome
        )
        for label in (self._retained, self._observed):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addWidget(QLabel("Current file evidence:"))
        self._evidence = QTextBrowser()
        self._evidence.setPlainText(
            "\n".join(f"{key}: {value}" for key, value in context.observed.evidence)
        )
        self._evidence.setFixedHeight(row_height() * 5)
        layout.addWidget(self._evidence)
        layout.addWidget(QLabel("Use instrument identity:"))
        self._choices = QComboBox()
        self._choices.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._choices.setMinimumContentsLength(40)
        initial = selection if selection is not None else retained or context.selection
        candidates = [item for item in (initial, retained) if item is not None]
        candidates.extend(context.observed.candidates)
        for profile in PROFILES:
            acquisitions = profile.detectors.acquisitions
            for acquisition in acquisitions:
                candidates.append(
                    InstrumentSelection(
                        profile.identity.key, acquisition.key, profile.identity.revision, "manual"
                    )
                )
            if not acquisitions:
                candidates.append(
                    InstrumentSelection(
                        profile.identity.key,
                        UnmappedAcquisition("undeclared"),
                        profile.identity.revision,
                        "manual",
                    )
                )
        candidates.append(GenericInstrumentIdentity())
        if selection is not None:
            candidates = [selection]
            self._choices.setEnabled(False)
        seen = set()
        if initial is None:
            self._choices.addItem("Choose an instrument identity…", None)
        for item in candidates:
            key = item.identity if isinstance(item, InstrumentSelection) else item
            if key not in seen:
                seen.add(key)
                self._choices.addItem(_identity_label(item), item)
        layout.addWidget(self._choices)
        self._consequence = QLabel()
        self._consequence.setTextFormat(Qt.TextFormat.PlainText)
        self._consequence.setWordWrap(True)
        layout.addWidget(self._consequence)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._choices.currentIndexChanged.connect(self._update_consequence)
        self._update_consequence()

    def _update_consequence(self):
        selected = self._choices.currentData()
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(selected is not None)
        message = "This changes identity only. Groups and calibrations are preserved; no calculation is run."
        if selected is not None and self._following is not None:
            updated = replace(
                self._run, instrument_choice=accept_instrument_choice(self._run, selected)
            )
            compatible = self._following.fingerprint.matches(profile_fingerprint_for_run(updated))
            message += (
                f' Keep following "{self._following.name}".'
                if compatible
                else f' This will stop following "{self._following.name}"; its base grouping and settings are retained.'
            )
        self._consequence.setText(message)

    def get_choice(self) -> RecordedInstrumentChoice:
        selected = self._choices.currentData()
        if selected is None:
            raise ValueError("An instrument identity must be selected before acceptance")
        return accept_instrument_choice(self._run, selected)
