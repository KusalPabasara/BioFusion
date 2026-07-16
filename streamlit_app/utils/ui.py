"""
Shared UI/theme for the BioFusion app — aqryl-inspired design system.

Adopts aqryl's clean, data-first language (primary blue #0066CC, black/white
neutrals, 8px spacing scale, Helvetica/Arial editorial + system-UI controls,
flat cards with subtle borders) — but with ONE deliberate deviation for this
clinical product: **no red**. Everywhere aqryl uses red (#DC2626) for
errors/loss, we use amber (#f59e0b), because in a medical result red reads as
alarm. Green here means "no signs found", amber means "attention", blue means
"informational / uncertain".
"""

import streamlit as st

ROLE_PATIENT = "Patient / Parent"
ROLE_CLINICIAN = "Clinician"

# --------------------------------------------------------------------------- #
# Global theme / CSS  (aqryl tokens)
# --------------------------------------------------------------------------- #
_GLOBAL_CSS = """
<style>
  :root {
    /* aqryl core */
    --primary: #0066CC;          /* primary blue — CTAs, active nav, links */
    --primary-hover: #0052A3;
    --primary-active: #003D7A;
    --primary-wash: rgba(0,102,204,0.10);
    --success: #059669;          /* gains / "no signs found" */
    --success-wash: rgba(5,150,105,0.10);
    /* no-red rule: amber stands in for aqryl's error red */
    --attention: #f59e0b;
    --attention-wash: rgba(245,158,11,0.10);

    --black: #000000;
    --white: #FFFFFF;
    --gray: #EFEFEF;             /* secondary bg, dividers, disabled */
    --ink-60: rgba(0,0,0,0.60);  /* helper text */
    --ink-75: rgba(0,0,0,0.75);
    --hair: #EFEFEF;
    --border-25: rgba(0,0,0,0.25);

    --font-ui: "Helvetica Neue", Helvetica, Arial, "Segoe UI", system-ui, sans-serif;
    --font-head: "Helvetica Neue", Helvetica, Arial, sans-serif;
    --radius-sm: 4px; --radius: 6px; --radius-md: 8px; --radius-lg: 12px;
    --shadow-md: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
  }

  html, body, [class*="css"] { font-family: var(--font-ui) !important; -webkit-font-smoothing: antialiased; }
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  .block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1080px; }

  /* Editorial headings use Helvetica, regular weight, tight leading (aqryl scale) */
  h1 { font-family: var(--font-head); font-size: 36px; font-weight: 400; line-height: 40px; color: var(--black); letter-spacing: 0; }
  h2 { font-family: var(--font-head); font-size: 30px; font-weight: 400; line-height: 41px; color: var(--black); }
  h3, h4 { font-family: var(--font-head); font-size: 16px; font-weight: 500; line-height: 22px; color: var(--black); }
  p, li, .stMarkdown { color: var(--black); }

  /* Primary buttons (aqryl) */
  .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
    background: var(--primary); color: var(--white); border: 0; border-radius: var(--radius-md);
    font-family: var(--font-ui); font-size: 16px; font-weight: 400; line-height: 24px;
    padding: 12px 24px; min-height: 44px;
  }
  .stButton > button[kind="primary"]:hover { background: var(--primary-hover); }
  .stButton > button[kind="primary"]:active { background: var(--primary-active); }
  /* Secondary buttons (transparent, 25% border) */
  .stButton > button[kind="secondary"] {
    background: transparent; color: var(--black); border: 1px solid var(--border-25);
    border-radius: var(--radius-md); font-family: var(--font-ui); font-size: 14px;
    line-height: 20px; min-height: 36px;
  }
  .stButton > button[kind="secondary"]:hover { background: var(--gray); border-color: rgba(0,0,0,0.5); }

  /* Nav pill row spacing */

  /* Badges / status chips (aqryl badge: 12px, 4px 8px, radius 4px) */
  .bf-badge { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 12px;
    padding: 4px 8px; border-radius: var(--radius-sm); font-weight: 500; }
  .bf-badge-primary { background: var(--primary-wash); color: var(--primary); }
  .bf-badge-success { background: var(--success-wash); color: var(--success); }
  .bf-badge-warning { background: var(--attention-wash); color: var(--attention); }

  /* Feature card (aqryl: flat, subtle border, md shadow, generous padding) */
  .bf-card { background: var(--white); border: 1px solid var(--hair); border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md); padding: 24px; }

  /* Result card */
  .bf-result { border-radius: var(--radius-lg); padding: 24px; margin: 8px 0 16px;
    border: 1px solid var(--hair); box-shadow: var(--shadow-md);
    animation: bf-rise 0.35s cubic-bezier(0.16,1,0.3,1); }
  @keyframes bf-rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  @media (prefers-reduced-motion: reduce) { .bf-result { animation: none; } }
  .bf-result .bf-label { display: inline-flex; align-items: center; gap: 0.5rem;
    font-weight: 500; font-size: 14px; letter-spacing: 0; text-transform: none; }
  .bf-result .bf-big { font-family: var(--font-head); font-size: 36px; font-weight: 400;
    line-height: 40px; margin: 12px 0 4px; }
  .bf-result .bf-sub { font-size: 12px; color: var(--ink-60); font-weight: 500; letter-spacing: 0; }
  .bf-result .bf-msg { margin-top: 16px; font-size: 14px; color: var(--black); line-height: 20px; }

  .bf-meter-track { width: 100%; height: 8px; background: var(--gray); border-radius: 5px; overflow: hidden; }
  .bf-meter-fill { height: 100%; border-radius: 5px; }

  /* Info panel (aqryl gray panel with left accent) */
  .bf-disclaimer { font-size: 12px; color: var(--ink-60); line-height: 16px;
    border-left: 4px solid var(--primary); padding: 8px 0 8px 12px; margin-top: 8px; }
</style>
"""


def inject_theme():
    """Inject the global stylesheet. Call once near the top of every page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def run_analysis_animation(stages, min_seconds_per_stage=0.55):
    """Play a staged 'analysing' animation while the pipeline runs.

    Inference is near-instant, but a short, honest walkthrough of the real steps
    (checking the image, correcting it, running the model, preparing guidance)
    reassures the user that the result was considered — important for trust in a
    clinical tool. `stages` is a list of (label, callable-or-None); each callable
    runs during its stage. Returns the list of callable results in order.

    Renders into a single placeholder that updates in place, then clears so the
    result can take over cleanly.
    """
    import time

    slot = st.empty()
    results = []
    n = len(stages)
    for i, (label, fn) in enumerate(stages):
        pct = int(round((i / n) * 100))
        _render_analysis_frame(slot, stages, i, pct)
        start = time.perf_counter()
        results.append(fn() if callable(fn) else None)
        # Ensure the stage is visible long enough to read (feels considered).
        elapsed = time.perf_counter() - start
        if elapsed < min_seconds_per_stage:
            time.sleep(min_seconds_per_stage - elapsed)
    _render_analysis_frame(slot, stages, n, 100)
    time.sleep(0.25)
    slot.empty()
    return results


def _render_analysis_frame(slot, stages, active_index, pct):
    """Draw one frame of the analysis animation into `slot`."""
    rows = []
    for j, (label, _) in enumerate(stages):
        if j < active_index:
            mark = ('<span style="color:var(--success);font-weight:700;">✓</span>')
            style = "color:var(--black);"
        elif j == active_index:
            mark = ('<span style="display:inline-block;width:14px;height:14px;'
                    'border:2px solid var(--primary);border-top-color:transparent;'
                    'border-radius:50%;animation:bf-spin 0.7s linear infinite;'
                    'vertical-align:-2px;"></span>')
            style = "color:var(--black);font-weight:600;"
        else:
            mark = ('<span style="display:inline-block;width:14px;height:14px;'
                    'border-radius:50%;border:2px solid var(--hair);"></span>')
            style = "color:var(--ink-60);"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;'
            f'font-size:0.95rem;{style}">{mark}<span>{label}</span></div>'
        )
    slot.markdown(f"""
    <style>@keyframes bf-spin {{ to {{ transform: rotate(360deg); }} }}</style>
    <div class="bf-card" style="max-width:420px;">
      <div style="font-weight:600;margin-bottom:12px;">Analysing X-ray…</div>
      <div class="bf-meter-track" style="margin-bottom:14px;">
        <div class="bf-meter-fill" style="width:{pct}%;background:var(--primary);
             transition:width 0.4s ease;"></div>
      </div>
      {''.join(rows)}
    </div>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    """A clean page header: a short accent bar + title + optional subtitle.

    No icon fonts — the accent bar carries the visual anchor (aqryl-minimal).
    """
    sub = (f'<p style="margin:2px 0 0; color:var(--ink-60); font-size:0.95rem;">'
           f'{subtitle}</p>') if subtitle else ""
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:1.4rem;">
      <span style="width:4px; height:38px; border-radius:2px; background:var(--primary);
            display:inline-block;"></span>
      <div>
        <h2 style="margin:0; font-size:1.6rem;">{title}</h2>
        {sub}
      </div>
    </div>
    """, unsafe_allow_html=True)


def top_nav(active: str = ""):
    """Render the shared top navigation bar. `active` = page key to highlight."""
    pages = [
        ("Home", "app.py"),
        ("Screening", "pages/1_Live_Prediction.py"),
        ("Model", "pages/2_Model_Insights.py"),
        ("Dataset", "pages/3_Dataset_Explorer.py"),
    ]
    cols = st.columns(len(pages))
    for col, (name, target) in zip(cols, pages):
        with col:
            kind = "primary" if name == active else "secondary"
            if st.button(name, use_container_width=True, type=kind, key=f"nav_{name}"):
                st.switch_page(target)


# --------------------------------------------------------------------------- #
# Role handling
# --------------------------------------------------------------------------- #
def get_role() -> str:
    return st.session_state.get("role", ROLE_PATIENT)


def role_selector(inline: bool = True):
    current = get_role()
    idx = 0 if current == ROLE_PATIENT else 1
    choice = st.radio(
        "I am a…", [ROLE_PATIENT, ROLE_CLINICIAN],
        index=idx, horizontal=inline, key="role_selector",
        help="Patients get plain-language guidance; clinicians get the full "
             "technical read-out.",
    )
    st.session_state["role"] = choice
    return choice


def is_clinician() -> bool:
    return get_role() == ROLE_CLINICIAN


# --------------------------------------------------------------------------- #
# Result card (role-aware)
# --------------------------------------------------------------------------- #
def render_result_card(triage_result, clinician: bool):
    """Render the triage result, adapting depth/tone to the audience."""
    t = triage_result
    color = t.color
    wash = {
        "#f59e0b": "rgba(245,158,11,0.10)",
        "#10b981": "rgba(5,150,105,0.10)",
        "#059669": "rgba(5,150,105,0.10)",
        "#2563eb": "rgba(0,102,204,0.10)",
        "#0066CC": "rgba(0,102,204,0.10)",
    }.get(color, "rgba(0,102,204,0.10)")
    pct = t.pneumonia_prob * 100
    st.markdown(f"""
    <div class="bf-result" style="background:{wash}; border-color:{color}33;">
        <div class="bf-label" style="color:{color};">
            <span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span>{t.label}
        </div>
        <div class="bf-big" style="color:{color};">{pct:.0f}%</div>
        <div class="bf-sub">PNEUMONIA PROBABILITY</div>
        <div class="bf-meter-track" style="margin-top:12px;">
            <div class="bf-meter-fill" style="width:{pct:.1f}%; background:{color};"></div>
        </div>
        <div class="bf-msg">{t.patient_message}</div>
    </div>
    """, unsafe_allow_html=True)

    if t.urgent:
        st.warning(f"**What to do next:** {t.patient_action}")
    else:
        st.info(f"**What to do next:** {t.patient_action}")

    if clinician:
        with st.container(border=True):
            st.markdown("**Clinical note**")
            st.markdown(
                f"{t.clinician_note}\n\n"
                f"- Pneumonia probability: **{t.pneumonia_prob:.3f}**\n"
                f"- Fail-safe decision threshold: **{t.threshold:.3f}**"
            )

    st.markdown(
        '<div class="bf-disclaimer">This tool is a <b>screening aid, not a '
        'diagnosis</b>. A qualified clinician must confirm any result before it '
        'informs care.</div>',
        unsafe_allow_html=True,
    )
