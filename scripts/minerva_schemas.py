"""
minerva_schemas.py
==================

Pydantic v2 data contracts for inter-script handoff in the
M.I.N.E.R.V.A. pipeline (scripts 13 -> 18 -> 21 -> 22 -> 23 -> 24 -> 25).

Why schemas:
  * The legacy pipeline passes raw dicts between scripts. A single
    script that drops or renames a key silently corrupts every
    downstream stage. The static-explanation problem and the
    pseudonym chaos both have this root cause: untyped handoffs.
  * Pydantic v2 gives us validation at script boundaries, JSON
    schema export for the Unity client, and clear error messages at
    thesis defence ("which stage produced this malformed card?").
  * This satisfies the white-box-testing requirement of §3.7.1 of
    the thesis: the Decision Layer's internal logic is now
    schema-checkable.

Refs: Khosravi et al. 2022 (auditability); Longo et al. 2024
(faithfulness boundary).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Verdict and indicator types
# ---------------------------------------------------------------------------
Verdict = Literal["FAKE", "REAL", "UNCERTAIN"]
DifficultyBin = Literal["easy", "medium", "hard"]
ExplanationTier = Literal["novice", "proficient", "advanced"]

# v2.6-final: CandidateCode used to be Literal["C-RM", "C-IB", "C-JS", "NONE"]
# but the v2.6-final editable config (scripts/candidate_config.py) lets
# the team rename codes (e.g. to C-A/C-B/C-C with common Filipino surnames
# per Roozenbeek 2020 fictional-examples principle). We relax this to a
# pattern-validated string so the schema picks up whatever codes the
# config currently specifies. The pattern still rejects malformed codes
# (must start with "C-" and contain only uppercase letters/dashes/digits)
# or "NONE". Validation against the configured candidates happens at
# the application layer (script 21's bucket logic, etc.).
CandidateCode = Annotated[
    str,
    Field(pattern=r"^(?:NONE|C-[A-Z0-9\-]{1,8})$"),
]

INDICATOR_CODES = [
    "EMO", "URG", "ANON", "MISS", "FAB", "POL",
    "CONS", "DISC", "IMP", "REV", "ENDO", "RECF",
]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class IndicatorDetail(BaseModel):
    """A single indicator's detection result for one card."""
    model_config = ConfigDict(extra="forbid")

    code: str
    label: str
    fired: bool
    evidence: list[str] = Field(default_factory=list, max_length=10)
    score: float = Field(ge=0.0, le=1.0)
    depict_family: str = ""


class QlatticeBlock(BaseModel):
    """QLattice symbolic-regression decision metadata."""
    model_config = ConfigDict(extra="forbid")

    score: float
    threshold: float = 0.5
    direction: str = ">="
    margin: float
    pred: int = Field(ge=0, le=1)
    equation: str = ""
    top_factors: list[dict[str, Any]] = Field(default_factory=list)


class DetectorBlock(BaseModel):
    """Probabilities from each underlying detector."""
    model_config = ConfigDict(extra="forbid")

    p_roberta_fake: float = Field(ge=0.0, le=1.0)
    p_distil_fake: float = Field(ge=0.0, le=1.0)
    p_degnn_fake: float = Field(ge=0.0, le=1.0)
    p_ensemble_fake: float = Field(ge=0.0, le=1.0)


class IndicatorPhrase(BaseModel):
    """One natural-language feedback line drawn from the response bank."""
    model_config = ConfigDict(extra="forbid")

    indicator: str = Field(description="indicator code, e.g. 'EMO'")
    phrase: str = Field(min_length=10, max_length=600)
    bank_ref: str = Field(description="bank slot id, e.g. 'EMO/v1/n3'")
    sift_move: str | None = None


class ExplanationBlock(BaseModel):
    """Student-facing explanation, content-aware and varied per-card."""
    model_config = ConfigDict(extra="forbid")

    tier: ExplanationTier
    summary: str = Field(min_length=10, max_length=1200)
    indicator_phrases: list[IndicatorPhrase] = Field(default_factory=list)
    sift_move: str | None = None
    credible_counter_card_id: str | None = None
    bank_version: str = "1.0"


class ProvenanceBlock(BaseModel):
    """Reproducibility & audit provenance attached to every card."""
    model_config = ConfigDict(extra="allow")  # allow downstream scripts to add keys

    seed: int
    git_sha: str = "unknown"
    bank_version: str = "1.0"
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_version: str = "2.0.0"
    script_chain: list[str] = Field(default_factory=list)


class ThemeFlags(BaseModel):
    """Output of script 23 theme classifier."""
    model_config = ConfigDict(extra="forbid")

    is_electoral: bool
    electoral_score: float = Field(ge=0.0, le=1.0)
    is_neutral_volume: bool = False
    classifier_label: str = "unknown"


# ---------------------------------------------------------------------------
# Top-level card
# ---------------------------------------------------------------------------
class UnityCard(BaseModel):
    """
    The single canonical card schema consumed by the Unity Chattr,
    VERIdict, VERIdex and VERITAS modules.

    Replaces the loose-dict format in the legacy unity_cards.json
    (which had 18+ ad-hoc top-level keys with no validation).
    """
    model_config = ConfigDict(extra="forbid")

    # --- Identity & content ---
    id: str
    candidate: CandidateCode
    text: str = Field(min_length=20, max_length=2000)
    target_label: Literal["fake", "real"]
    verdict: Verdict
    fake_likelihood_percent: float = Field(ge=0.0, le=100.0)
    credibility_percent: float = Field(ge=0.0, le=100.0)
    difficulty_bin: DifficultyBin

    # --- Indicators (the new pedagogy core) ---
    fired_indicators: list[str] = Field(default_factory=list)
    indicator_details: dict[str, IndicatorDetail] = Field(default_factory=dict)
    named_features: dict[str, float] = Field(default_factory=dict)

    # --- Detector outputs ---
    qlattice: QlatticeBlock
    detectors: DetectorBlock
    heuristics: dict[str, float] = Field(default_factory=dict)

    # --- Theme & explanation ---
    theme_flags: ThemeFlags
    explanation: ExplanationBlock

    # --- Provenance ---
    provenance: ProvenanceBlock
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- Validators ---
    @field_validator("fired_indicators")
    @classmethod
    def _validate_indicator_codes(cls, v: list[str]) -> list[str]:
        bad = [code for code in v if code not in INDICATOR_CODES]
        if bad:
            raise ValueError(f"Unknown indicator codes: {bad}")
        return v

    @field_validator("text")
    @classmethod
    def _validate_text_complete(cls, v: str) -> str:
        # v2.1: Lenient — only reject empty / degenerate text. The
        # truncation gate in script 13 already flags incomplete text;
        # the schema does not need to also reject it because real GPT-2
        # generations frequently lack a final period due to max_tokens
        # cutoff and are still pedagogically usable.
        v = v.strip()
        if not v:
            raise ValueError("Card text is empty.")
        if len(v) < 30:
            raise ValueError(f"Card text too short ({len(v)} chars).")
        return v

    @field_validator("id")
    @classmethod
    def _validate_id_pattern(cls, v: str) -> str:
        if not v or len(v) < 3:
            raise ValueError("card id too short")
        return v


class StoryCard(UnityCard):
    """A unity card promoted to the deck pool (v2.3) or daily-cycle deck.

    v2.3: `day` is OPTIONAL. The pool stores cards day-agnostic; the
    per-user draw script (28) assigns days at draw time. A drawn deck
    will have day=1..days; a pool card has day=None.
    """
    day: int | None = Field(default=None, ge=1, le=14)
    linked_blue_truth: str | None = None
    classification: str | None = None
    pool_index: int | None = None  # v2.3: position within the pool


class RejectionLog(BaseModel):
    """One row of the audit log for cards that failed any gate."""
    model_config = ConfigDict(extra="forbid")

    card_id: str
    stage: Literal[
        "theme_filter", "truncation_filter", "candidate_filter",
        "pseudonym_filter", "perplexity_filter", "indicator_filter",
        "schema_validation", "faithfulness_audit",
    ]
    verdict: Literal["reject", "warn"]
    reason: str
    seed: int | None = None
    git_sha: str | None = None
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateProfile(BaseModel):
    """VERIdex profile entry for one of the three fictional candidates."""
    model_config = ConfigDict(extra="forbid")

    code: CandidateCode
    name: str
    archetype: Literal["DYNASTIC", "REFORMIST", "POPULIST"]
    bio: str
    age: int = Field(ge=35, le=85)
    region: str
    party_acronym: str
    party_name: str
    platform_slogan: str
    policy_planks: list[str] = Field(min_length=3, max_length=8)
    indicator_susceptibility: dict[str, float]  # {code: 0..1 weighting}
    counter_narrative_anchors: list[str] = Field(min_length=2, max_length=6)
    references: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def validate_card_dict(d: dict) -> tuple[UnityCard | None, str | None]:
    """Try-validate a raw dict. Returns (card, error_msg)."""
    try:
        return UnityCard.model_validate(d), None
    except Exception as e:
        return None, str(e)


def export_jsonschema() -> dict:
    """Return the full JSON Schema for UnityCard (for Unity client)."""
    return UnityCard.model_json_schema()


if __name__ == "__main__":
    import json
    print(json.dumps(export_jsonschema(), indent=2)[:1500])
