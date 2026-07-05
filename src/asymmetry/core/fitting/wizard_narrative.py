"""Plain-language decision-trail narrative for the single-spectrum fit wizard.

This module is pure Python: no Qt, no matplotlib, no ``asymmetry.gui`` import.
Everything it produces is derived deterministically from a (possibly
deserialized) :class:`~asymmetry.core.fitting.fit_wizard.FitWizardRecommendation`
— never from live analysis state — so the trail persists for free with the
recommendation and an ``.asymp`` reload reproduces byte-identical prose. It is
window-agnostic: nothing here assumes a single dataset or a particular widget,
so the same builder can back both the single-spectrum wizard and a future
multi-dataset wizard.

Every accessor here is tolerant. A recommendation built from an old, partial,
or minimally-populated payload must still produce a full trail — degrading a
step to a simpler, honest sentence rather than raising or overstating what is
known. Nothing in this module ever asserts more certainty than the underlying
data support (see ``confidence_statement`` and the verdict wording in
``_verdict_step``).
"""

from __future__ import annotations

from dataclasses import dataclass

from asymmetry.core.fitting.fit_wizard import (
    ConfidenceTier,
    FitWizardRecommendation,
    RecommendationVerdict,
    _better_simpler_null,
    _template_family_map,
)

#: Plain-name / one-line physics gloss for each wizard candidate family key.
#: Keys match ``FamilyScreeningReport.family_key`` / ``WizardFamily.key``
#: exactly (see ``build_wizard_families`` in ``fit_wizard.py``).
FAMILY_GLOSSES: dict[str, tuple[str, str]] = {
    "relaxation": (
        "simple relaxation",
        "the muon spin relaxes at a single rate with no oscillation",
    ),
    "multi_rate": (
        "multi-rate relaxation",
        "more than one relaxation channel is superposed",
    ),
    "kt": (
        "static nuclear fields (Kubo-Toyabe)",
        "the muon sits in a static, randomly oriented local field from nearby nuclear moments",
    ),
    "oscillatory": (
        "precession signal",
        "the muon spin precesses in an internal or applied magnetic field",
    ),
    "muonium": (
        "muonium",
        "the muon has captured an electron, forming a hydrogen-like atom with hyperfine precession",
    ),
    "fmuf": (
        "muon-fluorine bonding (F-mu-F)",
        "the muon sits between two fluorine nuclei, giving a characteristic dipolar beat pattern",
    ),
    "baseline": (
        "current model",
        "the function already active in the fit tab, kept as a reference point",
    ),
}

#: Multiplet/envelope match ``kind`` values mapped to a short plain-physics
#: label for step 3 (see ``MultipletMatch.kind`` in ``peak_detection.py`` and
#: ``envelope_match.py`` for the authoritative list of kinds produced).
_MATCH_KIND_LABELS: dict[str, str] = {
    "larmor": "the muon Larmor frequency for the applied field",
    "muonium_low_tf": "a low-field muonium doublet",
    "muonium_high_tf": "a high-field muonium doublet",
    "muonium_zf": "a zero-field muonium triplet",
    "fmuf_linear": "a collinear muon-fluorine (F-mu-F) triplet",
    "muf": "a single-fluorine (mu-F) triplet",
    "fmuf_envelope": "a muon-fluorine (F-mu-F) time-domain signature",
    "muF_envelope": "a single-fluorine (mu-F) time-domain signature",
    "kt_envelope": "a static Kubo-Toyabe time-domain signature",
}

#: Confidence-tier wording, shared verbatim between the answer card and trail
#: step 6 (see ``confidence_statement``). Never edit these strings without
#: checking docs/reference/fit_wizard.rst, which quotes them verbatim.
_HIGH_STATEMENT = "High confidence — the recommended model describes the data cleanly."
_MEDIUM_STATEMENT = (
    "Medium confidence — this is the best model tried, but the fit leaves patterns "
    "in the residuals. Usable; review before publishing."
)
_NO_STRUCTURE_STATEMENT = (
    "The data look like a simple decay/flat background — no oscillation or extra "
    "structure is worth fitting."
)
_NONE_FALLBACK_STATEMENT = (
    "No confident recommendation could be formed from this analysis — inspect the "
    "comparison table before applying a model."
)


@dataclass(frozen=True)
class TrailStep:
    """One step of the decision trail.

    ``key`` and ``detail_kind`` are equal by construction (stable ids:
    ``"conditions"``, ``"families"``, ``"spectrum"``, ``"candidates"``,
    ``"verdict"``, ``"confidence"``) — a GUI maps ``detail_kind`` to the
    bespoke panel it re-parents for that step's inline expansion, and falls
    back to ``detail_lines`` verbatim when it has no bespoke panel for the
    key. ``headline`` is one plain sentence; ``detail_lines`` are short
    plain-text bullets suitable for the copy-log.
    """

    key: str
    headline: str
    detail_kind: str
    detail_lines: tuple[str, ...] = ()


def template_display_name(key: str | None, title: str) -> str:
    """Return a plain-physics display name for a template, without inventing physics.

    Looks the template key up in ``family_reports``-derived family membership
    (via the caller) is *not* done here — this function only glosses when a
    ``FAMILY_GLOSSES`` key is passed directly as ``key`` (i.e. the caller has
    already resolved family membership). If ``key`` is ``None``, unknown, or
    not a recognised family key, the raw ``title`` is returned unchanged —
    this deliberately covers null baselines (``null_constant``/``null_exp``)
    and peak-seeded multiplet templates (``oscillatoryN_..._constant``),
    neither of which appears in any family report.
    """
    if key is None:
        return title
    gloss = FAMILY_GLOSSES.get(key)
    if gloss is None:
        return title
    plain_name, _explanation = gloss
    if plain_name.lower() in title.lower():
        return title
    return f"{title} ({plain_name})"


def _family_key_for_template(template_key: str | None, family_map: dict[str, str]) -> str | None:
    if template_key is None:
        return None
    return family_map.get(template_key)


def _conditions_step(recommendation: FitWizardRecommendation) -> TrailStep:
    note = recommendation.scope_note.strip()
    if note:
        # ``note`` is already a complete, self-describing sentence fragment
        # (e.g. "run geometry: transverse field — screening TF families";
        # see ``wizard_scope.infer_auto_query``) — prefixing it with "Run
        # conditions:" would double up on "run" and read awkwardly, so it is
        # only capitalised and punctuated here, not re-framed.
        sentence = note[0].upper() + note[1:] + "."
        headline = sentence
        detail_lines = (sentence,)
    else:
        headline = "Run conditions were not recorded — every physics family was considered."
        detail_lines = (
            "No scope could be inferred from the run metadata, so no candidate "
            "family was excluded on that basis.",
        )
    return TrailStep(
        key="conditions",
        headline=headline,
        detail_kind="conditions",
        detail_lines=detail_lines,
    )


def _families_step(recommendation: FitWizardRecommendation) -> TrailStep:
    reports = recommendation.family_reports
    if not reports:
        return TrailStep(
            key="families",
            headline="No physics families were screened.",
            detail_kind="families",
            detail_lines=("No candidate family was in scope for this run.",),
        )
    screened_titles = [report.title for report in reports]
    headline = f"Physics families considered: {len(reports)} ({', '.join(screened_titles)})."
    detail_lines: list[str] = []
    for report in reports:
        gloss = FAMILY_GLOSSES.get(report.family_key)
        gloss_text = f" — {gloss[1]}" if gloss is not None else ""
        # Never echo ``report.reason`` here: promotion reasons routinely carry
        # technical/negative phrasing ("Stage-1 representative fit failed",
        # "gates failed") that would leak failure language into this step even
        # for a clean no-significant-structure result. This step reports only
        # titles, plain-physics glosses, and the promoted/not-promoted state.
        state = "expanded for detailed fitting" if report.promoted else "not expanded"
        detail_lines.append(f"{report.title}{gloss_text}: {state}.")
    return TrailStep(
        key="families",
        headline=headline,
        detail_kind="families",
        detail_lines=tuple(detail_lines),
    )


def _match_sentence(match_kind: str, note: str) -> str:
    """Return one complete plain sentence describing a pattern/envelope match.

    Known kinds get a fixed noun-phrase label composed into a single
    templated sentence. Unknown/future kinds fall back to the match's own
    human-readable ``note`` verbatim (already a full sentence fragment from
    ``peak_detection.py``/``envelope_match.py``) rather than composing a
    frame around it, since that note may already contain "matches" and a
    second composed frame would read as doubled prose ("shape matches
    matches ...").
    """
    label = _MATCH_KIND_LABELS.get(match_kind)
    if label:
        return f"Detected {label}."
    return note.strip() or "An unlabelled pattern match was detected."


def _spectrum_step(recommendation: FitWizardRecommendation) -> TrailStep:
    analysis = recommendation.peak_analysis
    matches = recommendation.multiplet_matches
    peak_count = len(analysis.peaks) if analysis is not None else 0

    if peak_count == 0 and not matches:
        headline = "Spectral search found no significant spectral lines or patterns."
        detail_lines = (
            "No spectral lines cleared the detection threshold, and no known "
            "physics pattern was matched in the time domain.",
        )
        return TrailStep(
            key="spectrum", headline=headline, detail_kind="spectrum", detail_lines=detail_lines
        )

    headline = f"Spectral search: {peak_count} line(s) detected, {len(matches)} pattern match(es)."
    detail_lines = [f"{peak_count} spectral line(s) detected." if analysis is not None else ""]
    detail_lines = [line for line in detail_lines if line]
    for match in matches:
        sentence = _match_sentence(match.kind, match.note).rstrip(".")
        detail_lines.append(f"{sentence} (match quality {match.quality:.2f}).")
        derived_bits = ", ".join(f"{name} = {value:.4g}" for name, value in match.derived_values)
        if derived_bits:
            detail_lines.append(f"  Derived: {derived_bits}.")
    if not matches:
        detail_lines.append("No known physics pattern matched the detected lines.")
    return TrailStep(
        key="spectrum",
        headline=headline,
        detail_kind="spectrum",
        detail_lines=tuple(detail_lines),
    )


#: Plain-condensed categories for disqualification/rejection reasons. Raw
#: reason strings from ``fit_wizard.py`` (residual-gate reasons, disqualifier
#: reasons) are never echoed verbatim here — they contain technical/negative
#: language ("Fit failed", "gates failed") that would read as an error report
#: rather than a curated result, and would defeat the "no failure language in
#: the null-structure result" contract. Matched by substring, longest/most
#: specific first.
_REJECTION_PHRASE_RULES: tuple[tuple[str, str], ...] = (
    ("consistent with zero", "the fitted oscillation amplitude was indistinguishable from zero"),
    ("resolution floor", "the fitted oscillation had no support in the spectrum"),
    ("pinned at its", "a fitted parameter ran to the edge of its allowed range"),
    (
        "no supporting detected spectral peak",
        "the fitted oscillation had no support in the spectrum",
    ),
    ("completes only", "the fitted oscillation had no support in the spectrum"),
)


def _condense_disqualification_reasons(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return "did not meet the wizard's quality checks"
    phrases: list[str] = []
    for reason in reasons:
        matched = next(
            (phrase for needle, phrase in _REJECTION_PHRASE_RULES if needle in reason),
            None,
        )
        phrases.append(matched or "did not meet the wizard's quality checks")
    # De-duplicate while preserving order.
    seen: list[str] = []
    for phrase in phrases:
        if phrase not in seen:
            seen.append(phrase)
    return "; ".join(seen)


def _candidates_step(recommendation: FitWizardRecommendation) -> TrailStep:
    assessments = recommendation.assessments
    if not assessments:
        return TrailStep(
            key="candidates",
            headline="No candidate models were fitted.",
            detail_kind="candidates",
            detail_lines=("The analysis did not fit any candidate model.",),
        )

    fitted = [a for a in assessments if not a.is_null_baseline]
    successful = [a for a in fitted if a.is_successful]
    nulls = [a for a in assessments if a.is_null_baseline]
    disqualified = [a for a in successful if a.is_disqualified]
    promoted_count = sum(1 for report in recommendation.family_reports if report.promoted)
    family_count = len(recommendation.family_reports)

    headline = (
        f"{len(successful)} candidate model(s) fitted successfully"
        f"{f' ({len(nulls)} reference baseline(s) also fitted)' if nulls else ''}."
    )
    detail_lines = [
        f"{len(fitted)} candidate model(s) attempted; {len(successful)} fitted successfully.",
    ]
    if family_count:
        detail_lines.append(
            f"{promoted_count} of {family_count} physics families were expanded for "
            "detailed fitting."
        )
    if nulls:
        detail_lines.append(f"{len(nulls)} reference (null) baseline(s) fitted for comparison.")
    if disqualified:
        condensed = _condense_disqualification_reasons(
            tuple(
                reason
                for assessment in disqualified
                for reason in assessment.disqualification_reasons
            )
        )
        detail_lines.append(f"{len(disqualified)} rejected: {condensed}.")
    else:
        detail_lines.append("No candidates were rejected on quality grounds.")
    return TrailStep(
        key="candidates",
        headline=headline,
        detail_kind="candidates",
        detail_lines=tuple(detail_lines),
    )


def _decisiveness_phrase(recommendation: FitWizardRecommendation) -> str:
    """Return a truthful decisiveness clause, or "" when it cannot be computed.

    Only claims a margin when the winning assessment and a strictly-simpler
    null baseline are both *present* in ``recommendation.assessments`` — on a
    legacy/deserialized payload missing the null assessments, this degrades to
    "" (the caller then falls back to plain structural wording) rather than
    inventing a number. Deliberately coarse (large vs. not) rather than a
    finer band, which would imply precision the AICc margin does not carry in
    plain language.
    """
    winner = recommendation.recommended_assessment
    if winner is None:
        return ""
    null_assessments = [
        a for a in recommendation.assessments if a.is_null_baseline and a.is_successful
    ]
    if not null_assessments:
        return ""
    reference_null = _better_simpler_null(winner, null_assessments, recommendation.metric)
    if reference_null is None:
        return ""
    delta = reference_null.metric_value(recommendation.metric) - winner.metric_value(
        recommendation.metric
    )
    if not (delta == delta):  # NaN guard without importing math/numpy here.
        return ""
    if delta >= 10.0:
        return " clearly better than a plain relaxation baseline"
    return ""


def _verdict_step(recommendation: FitWizardRecommendation) -> TrailStep:
    verdict = recommendation.verdict
    winner = recommendation.recommended_assessment
    family_map = _template_family_map(recommendation.family_reports)

    if verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE:
        headline = _NO_STRUCTURE_STATEMENT
        detail_lines = [_NO_STRUCTURE_STATEMENT]
        if recommendation.caveat:
            detail_lines.append(recommendation.caveat)
        return TrailStep(
            key="verdict",
            headline=headline,
            detail_kind="verdict",
            detail_lines=tuple(detail_lines),
        )

    if verdict is RecommendationVerdict.NONE or winner is None:
        headline = "No recommendation could be formed from this analysis."
        detail_lines = [headline]
        if recommendation.summary:
            detail_lines.append(recommendation.summary)
        return TrailStep(
            key="verdict",
            headline=headline,
            detail_kind="verdict",
            detail_lines=tuple(detail_lines),
        )

    family_key = _family_key_for_template(winner.template.key, family_map)
    display_name = template_display_name(family_key, winner.template.title)
    decisiveness = _decisiveness_phrase(recommendation)
    if decisiveness:
        headline = f"Best model: {display_name} —{decisiveness}."
    else:
        headline = f"Best model: {display_name} — describes real structure in the data."
    detail_lines = [headline]
    if recommendation.comparable_keys:
        detail_lines.append("A similarly scoring alternative model is available to compare.")
    return TrailStep(
        key="verdict", headline=headline, detail_kind="verdict", detail_lines=tuple(detail_lines)
    )


def confidence_statement(recommendation: FitWizardRecommendation) -> str:
    """Return the shared confidence/verdict wording for the card and step 6.

    Wording is fixed verbatim per tier (see the module-level ``_..._STATEMENT``
    constants) so the answer card and the trail's confidence step never
    disagree. Only **Medium** confidence appends the specific caveat text to
    the statement itself — the wording map reserves that for Medium alone.
    The ``NONE`` + ``NO_SIGNIFICANT_STRUCTURE`` result framing is returned
    bare (no appended caveat): it is the shared card string, and appending a
    numeric ΔAICc caveat would put jargon on a plain-language card. Any other
    ``NONE`` gets an honest fallback. Callers that want the caveat as extra
    guidance (e.g. the trail's confidence-step detail bullets) should read
    ``recommendation.caveat`` directly.
    """
    if recommendation.confidence is ConfidenceTier.HIGH:
        return _HIGH_STATEMENT
    if recommendation.confidence is ConfidenceTier.MEDIUM:
        caveat = recommendation.caveat.strip()
        return f"{_MEDIUM_STATEMENT} {caveat}".strip() if caveat else _MEDIUM_STATEMENT
    # confidence is NONE.
    if recommendation.verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE:
        return _NO_STRUCTURE_STATEMENT
    return _NONE_FALLBACK_STATEMENT


def _confidence_step(recommendation: FitWizardRecommendation) -> TrailStep:
    statement = confidence_statement(recommendation)
    detail_lines = [statement]
    # The caveat is guidance, not card wording: it belongs in the step's
    # detail bullets (the copy-log, the expanded panel) even when
    # ``confidence_statement`` keeps the shared card string bare (the
    # NONE + NO_SIGNIFICANT_STRUCTURE case).
    caveat = recommendation.caveat.strip()
    if caveat and recommendation.confidence is not ConfidenceTier.MEDIUM:
        detail_lines.append(caveat)
    return TrailStep(
        key="confidence",
        headline=statement,
        detail_kind="confidence",
        detail_lines=tuple(detail_lines),
    )


def build_wizard_trail(recommendation: FitWizardRecommendation) -> tuple[TrailStep, ...]:
    """Derive the six-step decision trail from a fit-wizard recommendation.

    Every field access is tolerant of missing/``None``/empty data — a step
    degrades to a simpler, honest sentence rather than raising. The trail is a
    pure function of ``recommendation``, so it reproduces identically after an
    ``.asymp`` reload or a serialization round trip.
    """
    return (
        _conditions_step(recommendation),
        _families_step(recommendation),
        _spectrum_step(recommendation),
        _candidates_step(recommendation),
        _verdict_step(recommendation),
        _confidence_step(recommendation),
    )


def render_log_text(recommendation: FitWizardRecommendation) -> str:
    """Render the full decision trail as plain text for a copy-log / bug report."""
    lines: list[str] = []
    for step in build_wizard_trail(recommendation):
        lines.append(step.headline)
        for detail in step.detail_lines:
            lines.append(f"  - {detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
