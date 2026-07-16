"""
Clinical decision logic: triage banding, escalation guidance, and an honest
"uncertain" state.

The raw model output is a pneumonia probability. On its own that is not safe to
show a patient. This module turns it into:

  * a triage BAND (clear / likely / uncertain / normal) using both the model's
    fail-safe decision threshold and an uncertainty margin around it,
  * ESCALATION guidance appropriate to the band and the audience (patient vs
    clinician),

so the same prediction is communicated safely and consistently everywhere.

Nothing here is a diagnosis — every message frames the result as a screening aid
that requires clinician confirmation.
"""

from dataclasses import dataclass

# Half-width of the "uncertain" zone around the decision threshold. Only images
# whose pneumonia probability sits genuinely *on the fence* (within this margin
# of the threshold) get the "uncertain — get a second opinion" result; clearly
# low probabilities read as reassuring "no clear signs". Kept narrow so a
# confidently-low score (e.g. 0.07) isn't needlessly flagged uncertain.
UNCERTAIN_MARGIN = 0.04

# Bands
BAND_PNEUMONIA_CLEAR = "pneumonia_clear"
BAND_PNEUMONIA_LIKELY = "pneumonia_likely"
BAND_UNCERTAIN = "uncertain"
BAND_NORMAL = "normal"


@dataclass
class Triage:
    band: str
    label: str            # short human label
    pneumonia_prob: float
    threshold: float
    color: str            # palette hex (no red — project rule)
    patient_message: str  # plain-language, for patients/parents
    patient_action: str   # what the patient should do next
    clinician_note: str   # concise note for clinicians
    urgent: bool          # should this be surfaced/escalated promptly


# Palette — aqryl tokens, no red (project rule). Amber = attention/pneumonia,
# green = reassuring/normal, blue = informational/uncertain.
_AMBER = "#f59e0b"      # aqryl's error-red slot, substituted with amber
_EMERALD = "#059669"    # aqryl success green
_SAPPHIRE = "#0066CC"   # aqryl primary blue


def triage(pneumonia_prob: float, threshold: float = 0.5) -> Triage:
    """Map a pneumonia probability to a triage band + guidance.

    Args:
        pneumonia_prob: model probability of the PNEUMONIA class in [0,1].
        threshold: the fail-safe decision threshold (from training).
    """
    p = float(pneumonia_prob)

    # Safety invariant (defence in depth): a pneumonia probability at or above
    # 0.5 must NEVER be shown as the reassuring green "normal" band, regardless
    # of the tuned threshold. If the model leans pneumonia, the user must see an
    # attention/urgent result — never a green all-clear. This guards against a
    # mis-tuned threshold ever producing a dangerous false reassurance.
    if p >= 0.5 and p < threshold:
        return Triage(
            band=BAND_PNEUMONIA_LIKELY,
            label="Possible pneumonia",
            pneumonia_prob=p, threshold=threshold, color=_AMBER,
            patient_message="This X-ray may show signs of pneumonia.",
            patient_action=(
                "Please see a doctor to confirm. This is a screening aid, not a "
                "diagnosis."
            ),
            clinician_note=(
                "Pneumonia probability >= 0.5 though below the tuned fail-safe "
                "threshold; surfaced as attention rather than normal by design."
            ),
            urgent=True,
        )

    near = abs(p - threshold) <= UNCERTAIN_MARGIN

    if near:
        return Triage(
            band=BAND_UNCERTAIN,
            label="Uncertain",
            pneumonia_prob=p, threshold=threshold, color=_SAPPHIRE,
            patient_message=(
                "The result is borderline — the tool can't tell confidently "
                "whether this X-ray shows pneumonia."
            ),
            patient_action=(
                "Please see a doctor for a proper reading. Do not rely on this "
                "screening result alone."
            ),
            clinician_note=(
                "Probability sits within the uncertainty margin of the decision "
                "threshold; recommend radiologist review rather than an automated call."
            ),
            urgent=True,
        )

    if p >= threshold:
        # Above threshold — split into "likely" vs "clear" by how far above.
        clear = p >= min(threshold + 0.30, 0.90)
        return Triage(
            band=BAND_PNEUMONIA_CLEAR if clear else BAND_PNEUMONIA_LIKELY,
            label="Signs of pneumonia" if clear else "Possible pneumonia",
            pneumonia_prob=p, threshold=threshold, color=_AMBER,
            patient_message=(
                "This X-ray shows signs consistent with pneumonia."
                if clear else
                "This X-ray may show early or mild signs of pneumonia."
            ),
            patient_action=(
                "Seek medical care promptly and bring this X-ray with you. This "
                "is a screening result, not a diagnosis."
                if clear else
                "Please see a doctor soon to confirm. This is a screening aid, "
                "not a diagnosis."
            ),
            clinician_note=(
                "Above fail-safe threshold; sensitivity-prioritised. Confirm with "
                "clinical correlation and radiology."
            ),
            urgent=clear,
        )

    # Below threshold — reassuring, but still not a clean bill of health.
    return Triage(
        band=BAND_NORMAL,
        label="No clear signs",
        pneumonia_prob=p, threshold=threshold, color=_EMERALD,
        patient_message=(
            "This X-ray does not show clear signs of pneumonia to the tool."
        ),
        patient_action=(
            "If you feel unwell or symptoms continue, still see a doctor — this "
            "tool can miss cases and is not a diagnosis."
        ),
        clinician_note=(
            "Below fail-safe threshold. Note the tool prioritises sensitivity; "
            "specificity is lower, so false-positives are expected upstream."
        ),
        urgent=False,
    )
