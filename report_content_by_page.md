# PNEUMOSCAN AI — UPDATED REPORT CONTENT (mapped to your 9-page layout)

Copy each block into the matching page of your existing report.
NEW pages required by the semifinal guidelines are marked ★ NEW PAGE.

# ======================================================================
PAGE 1 — COVER  (keep your BRAINSTORM cover design; update the text)

Title:
    PNEUMONIA DETECTION FROM CHEST X-RAYS USING DEEP LEARNING
    (Product: PneumoScan AI)

Subtitle (add this line):
    A live, explainable, fail-safe screening tool — deployed at pneu.gmora.dev

Team GMora — University of Moratuwa
Kusal Pabasara · Pasindu Mihiranga · Thanoj Buddhima · Senuja Dilmith

(Optional badges to add on the cover:)
    Live at pneu.gmora.dev   ·   94.2% accuracy · 0.988 AUC   ·   Installable mobile app

# ======================================================================
PAGE 2 — OUR TEAM  (keep as-is — no changes needed)

Keep your existing team member details page unchanged.

# ======================================================================
PAGE 3 — 1. THE PROBLEM BEING ADDRESSED AND WHY IT IS IMPORTANT

Pneumonia is one of the leading causes of mortality worldwide, disproportionately
affecting children under five and elderly adults. In Sri Lanka, respiratory infections
remain among the top causes of hospital admissions and paediatric deaths, yet accurate
and timely diagnosis is hampered by a critical shortage of radiologists, particularly in
rural and district-level health facilities.

GLOBAL AND LOCAL STATISTICS
• ~450 million pneumonia cases occur globally every year (WHO).
• ~2.5 million deaths were attributed to pneumonia recently, making it the leading
  infectious killer of children under 5 worldwide.
• Pneumonia accounts for approximately 15% of all deaths in children under five.
• In Sri Lanka, rural hospitals and district-level centres face a severe radiologist
  shortage, leading to diagnostic delays of 24–72 hours.
• Existing workflows are manual, inconsistent, and unavailable around the clock, leaving
  night-time and weekend emergencies inadequately supported.

UNMET CLINICAL NEEDS
• Radiologist Shortage: chest X-rays go unread or delayed in rural Sri Lanka.
• Diagnostic Variability: manual interpretation varies between observers, especially for
  subtle or early-stage pneumonia.
• No Affordable AI Pre-Screening: no low-cost AI triage tool is widely deployed in Sri
  Lankan hospitals.
• 24/7 Unavailability: radiology services are rarely continuous, leaving gaps in emergency
  care.

This aligns with the Brainstorm 2026 theme, "Innovative and Affordable Healthcare Solutions
for Local Challenges." An affordable AI-powered screen deployed in rural clinics and
district hospitals can prevent hundreds of avoidable deaths annually.

# ======================================================================
PAGE 4 — 2. PROCESS OF INFORMATION COLLECTION  +  3. THE SOLUTION (start)

1. PROCESS OF INFORMATION COLLECTION ABOUT THE PROBLEM

• WHO & Global Health Data — referenced WHO statistics and global reports to quantify the
  burden of pneumonia globally and in South Asia.
• Peer-reviewed Clinical Dataset — Kermany et al. (2018, Cell), sourced from Guangzhou
  Women and Children's Medical Centre, validating the authenticity of the training data.
• Adult Dataset — the COVID-19 Radiography Database was added to extend coverage to adult
  chest X-rays.
• Expert Feedback — our team participated in the IEEE EMBS BioFusion Hackathon 2026, where
  domain experts reviewed the project and guided refinement of the model and app design.

1. BRIEF DESCRIPTION OF THE SOLUTION

OVERVIEW
PneumoScan AI is a live web and mobile application (deployed at pneu.gmora.dev) that lets a
healthcare worker — or a patient — capture or upload a chest X-ray and instantly receive an
AI-powered analysis. The app shows a clear pneumonia risk percentage, a colour-coded
Grad-CAM heatmap highlighting the lung regions that influenced the prediction, and plain
next-step guidance. This makes the AI decision transparent, clinically interpretable, and
immediately actionable — even by a general practitioner or nurse in a rural clinic with no
radiologist. It is installable to a phone home screen (PWA) and works over a secure HTTPS
connection.

HOW THE APP WORKS — STEP BY STEP

1. Capture or Upload — take a photo of the X-ray with a phone camera, or upload a digital
  file (JPEG/PNG). A role selector tailors the experience for a Patient/Parent or a Clinician.
2. Quality Gate — an automatic check rejects unusable images and, for phone photos,
  tolerates the soft focus and warm colour cast normal for a snapshot of a film.
3. Phone-Photo Rectification — for photos of physical films, OpenCV detects the film boundary
  and corrects the camera-angle perspective.
4. AI Analysis — a fine-tuned ResNet-50 outputs a calibrated pneumonia probability, converted
  into a triage band by a fail-safe decision threshold.
5. Explainable Result — a Grad-CAM heatmap shows which lung regions drove the prediction,
  alongside plain-language guidance and a "screening aid, not a diagnosis" disclaimer.

# ======================================================================
PAGE 5 — RESULT COMMUNICATION  +  TWO AUDIENCES  (fills your page 5)

RESULT COMMUNICATION — SAFE BY DESIGN
• "No clear signs"        → low pneumonia probability → reassurance + "see a doctor if
                            symptoms persist."
• "Uncertain"             → borderline / on the fence → "get a second opinion" (never a
                            confident call).
• "Signs of pneumonia"    → above the fail-safe threshold → prompt to seek clinical care
                            promptly.

TWO AUDIENCES, ONE ENGINE
• For Patients & Parents — plain-language result, clear next-step guidance, and a strong
  prompt to consult a clinician. No jargon.
• For Clinicians — Grad-CAM overlay, class probabilities, the fail-safe operating point, and
  full technical detail for decision support.

POSITIVE RESULT (VERIFIED)
On a real phone photo of an adult pneumonia X-ray from a Moratuwa clinic, the system
correctly returns "Signs of pneumonia"; on a healthy adult phone X-ray it returns "No clear
signs" — the two cases the earlier pediatric-only model could not handle.

# ======================================================================
PAGE 6 — MODEL ARCHITECTURE AND PERFORMANCE  +  PRODUCT ARCHITECTURE

MODEL ARCHITECTURE AND TRAINING STRATEGY
We use ResNet-50 (ImageNet-pretrained) and now fine-tune the deepest residual stage (layer4)
plus the classifier head — about 15 million trainable parameters — so high-level features
adapt from natural images to radiographs. The model is trained on a COMBINED corpus of
23,405 images: pediatric (Kermany/Guangzhou, 5,856) plus adult (COVID-19 Radiography
Database, 17,549), near class-balanced (~50/50). Training uses AdamW with weight decay, label
smoothing, a cosine learning-rate schedule, and early stopping. A dedicated phone-photo
augmentation (perspective, warm colour cast, glare, vignette, blur, JPEG) makes the model
robust to real phone captures of films. The decision threshold is tuned on a validation set
for ≥95% sensitivity and capped so the tool errs toward flagging pneumonia.

HELD-OUT TEST PERFORMANCE (3,510 images)
    Accuracy .................. 94.2%
    Sensitivity (recall) ...... 94.3%
    Specificity ............... 94.2%
    Precision ................. 94.1%
    F1-score .................. 0.942
    AUC-ROC ................... 0.988

IMPROVEMENT OVER THE PREVIOUS MODEL
    Metric                 | Old (pediatric only) | Now (combined + phone-trained)
    Test accuracy          | 86.7%                | 94.2%
    AUC-ROC                | 0.943                | 0.988
    Sensitivity/Prec./F1   | 96.7% / 84.3% / 0.90 | 94.3% / 94.1% / 0.94
    Works on adult X-rays  | No                   | Yes — 92.4% recall, AUC 0.984
    Works on phone photos  | No                   | Yes — verified on real captures
    Overfitting gap        | 7.7%                 | 1.1%

VALIDATION DEPTH
Beyond a single split, we ran 5-fold stratified cross-validation (sensitivity 98.7% ± 0.5%
across folds) and report per-population results — pediatric (97.6% recall, AUC 0.995) and
adult (92.4% recall, AUC 0.984) — showing the model generalises across ages.

PRODUCT ARCHITECTURE (inference pipeline)
Capture/Upload → Quality gate (OpenCV) → Phone-photo rectification → Preprocess (224×224, RGB,
ImageNet normalisation) → ResNet-50 (fine-tuned) → Softmax P(Normal)/P(Pneumonia) →
Fail-safe triage band → Grad-CAM heatmap → Role-aware result display.

# ======================================================================
PAGE 7 — REQUIRED RESOURCES  +  NOVELTY  +  MARKET ANALYSIS

REQUIRED RESOURCES / TECHNOLOGY STACK (all in use)
• Model: PyTorch 2.6, torchvision ResNet-50 (trained on an NVIDIA RTX 3050 GPU; runs
  CPU-only inference in production).
• Vision utilities: OpenCV, NumPy, Pillow (quality checks, rectification, Grad-CAM).
• App/UI: Streamlit with custom CSS and PWA (camera capture, installable home-screen app).
• Server: AWS EC2 (Ubuntu) with nginx and systemd; HTTPS via Let's Encrypt.
• Data: 23,405 openly licensed, peer-reviewed images (Kermany + COVID-19 Radiography DB).
• Edge option: Raspberry Pi / NVIDIA Jetson Nano for fully offline use in rural clinics.

NOVELTY AND UNIQUENESS
• Deployed, not just a concept — a live, installable app with camera capture, unlike most
  academic prototypes.
• Explainable AI — Grad-CAM heatmaps make every prediction transparent, building clinical
  trust.
• Fail-safe, sensitivity-first design — the threshold is tuned to minimise missed pneumonia,
  with a calibrated "uncertain" band for borderline cases.
• Robust to real inputs — validated on adult anatomy AND phone photos of films, not just
  clean digital scans.
• Affordable & locally tailored — runs on free/low-cost infrastructure; designed for Sri
  Lankan workflows with planned bilingual (Sinhala/English) interfaces and eSuwa integration.

MARKET ANALYSIS AND COMPETITORS
Global competitors include Qure.ai (India), Lunit INSIGHT CXR (South Korea), and Aidoc (USA).
These are enterprise-grade, expensive solutions priced out of reach for rural Sri Lankan
clinics. No affordable, locally-tailored, open-source pneumonia screening app currently
serves the Sri Lankan market, making PneumoScan AI a novel and impactful entry.

# ======================================================================
PAGE 8 — 4. IMPACT ON THE HEALTHCARE SECTOR  +  BENEFITS  +  SUSTAINABILITY

ADDRESSING THE PROBLEM DIRECTLY
PneumoScan AI turns the current 24–72 hour diagnostic delay into a sub-30-second,
always-available assessment through a simple smartphone or web upload, directly resolving the
three unmet needs: radiologist shortage, diagnostic delay, and the absence of automated
pre-screening.

BENEFITS TO STAKEHOLDERS
• Rural Health Workers — rapid AI-assisted pre-screening on a basic smartphone; faster
  decisions without specialist access.
• Patients — shorter waits, fewer missed cases, earlier treatment, improved outcomes.
• District Hospitals — a 24/7 second opinion that helps prioritise urgent cases and reduce
  radiologist load.
• Ministry of Health — scalable, low-cost deployment and a basis for national pneumonia-trend
  tracking.

SUSTAINABILITY & COST-EFFECTIVENESS
Built entirely on open-source tools and freely available datasets, with CPU-only inference on
a single low-cost cloud VM and zero per-query licensing — making per-scan cost negligible
versus enterprise tools. A freemium model for NGOs and government hospitals ensures broad
access, while a subscription tier for private hospitals funds ongoing maintenance. The model
improves as partner hospitals contribute regional X-ray data.

# ======================================================================
PAGE 9 — SCALABILITY AND FUTURE ROADMAP

• Phase 1 (Done): Deployed binary pneumonia/normal screening with Grad-CAM explanation, a
  fail-safe threshold, and robustness to adult, pediatric, and phone-photo inputs.
• Phase 2: Multi-class classification distinguishing bacterial vs. viral pneumonia for more
  targeted treatment guidance.
• Phase 3: Expansion to additional chest pathologies common in Sri Lanka (tuberculosis,
  pleural effusion).
• Phase 4: Regulatory pathway as a Software as a Medical Device (SaMD) — registration with
  NMRA Sri Lanka, supported by a local clinical-validation study.
• Phase 5: Integration with eSuwa and national health IT systems, and bilingual
  (Sinhala/English) interfaces.

SOCIAL IMPACT
By making pneumonia screening accessible, fast, and affordable across rural Sri Lanka,
PneumoScan AI can prevent hundreds of avoidable deaths annually — bringing specialist-level
decision support to the settings that need it most.

######################################################################
★ NEW PAGES — ADD THESE (required by the semifinal guidelines &
  evaluation criteria; your current report does not have them)
######################################################################

# ======================================================================
★ NEW PAGE 10 — LITERATURE REVIEW AND BACKGROUND

• Rajpurkar et al. (2017), CheXNet — a 121-layer DenseNet reaching radiologist-level
  pneumonia detection. We add explainability and a deployed, fail-safe workflow for
  low-resource use.
• Kermany et al. (2018) — transfer learning on an expert-graded pediatric CXR dataset. We
  generalise beyond pediatrics to adults and phone photos.
• Selvaraju et al. (2017), Grad-CAM — gradient-based visual explanations for CNNs, integrated
  into every prediction to build clinical trust.
• He et al. (2016), ResNet — residual learning enabling deep networks; our fine-tuned
  backbone.
• Phillips et al., CheXphoto — a photographed-CXR robustness benchmark; we reproduce capture
  artefacts synthetically to train for phone use.

GAP ADDRESSED: most published models are accurate on clean, in-distribution scans but are not
deployment-ready, not explainable, and not validated on the messy inputs (adult anatomy,
phone photos) a real rural workflow produces. PneumoScan AI targets exactly that gap.

# ======================================================================
★ NEW PAGE 11 — FEASIBILITY

• Technical feasibility — built on mature open-source frameworks (PyTorch, OpenCV,
  Streamlit); the ~90 MB model runs in under a second on CPU, no GPU needed in production.
• Operational feasibility — deployed on a single low-cost cloud VM with automated TLS and
  process supervision; an edge option removes the internet dependency for remote clinics.
• Data feasibility — trained on 23,405 openly licensed, peer-reviewed images; the pipeline
  can ingest new regional data as partner hospitals contribute.
• Economic feasibility — zero per-query licensing; free-tier/low-cost infrastructure makes
  per-scan cost negligible.

# ======================================================================
★ NEW PAGE 12 — REGULATORY ASPECTS AND INTELLECTUAL PROPERTY

PneumoScan AI is positioned as Software as a Medical Device (SaMD) and framed explicitly as a
screening decision-support aid, NOT a diagnostic device — keeping a clinician in the loop and
lowering the regulatory risk class.

• NMRA (Sri Lanka) — primary regulator. We target registration as a low–moderate risk SaMD,
  with a documented clinical-validation study on local X-rays before any clinical deployment.
• US FDA — reference framework: an equivalent product would pursue a 510(k) pathway as Class
  II. We follow FDA "Good Machine Learning Practice" and SaMD guidance for design controls.
• Standards — aligning with IEC 62304 (medical device software lifecycle) and ISO 14971 (risk
  management).

• The trained model, phone-photo augmentation pipeline, fail-safe threshold logic, and the
  patient/clinician application are our original contributions.
• All third-party components (ResNet-50 weights, datasets, libraries) are used under
  permissive open-source / open-access licences, documented for compliance.
• Strategy: retain proprietary rights over the application and locally-trained weights while
  keeping the core screening accessible to public hINTELLECTUAL PROPERTY
ospitals.

# ======================================================================
★ NEW PAGE 13 — DATA SECURITY AND PRIVACY  +  ETHICAL REVIEW

DATA SECURITY AND PRIVACY POLICY
• In transit — all traffic is encrypted over HTTPS (TLS); camera access is granted only on a
  secure origin.
• Data minimisation — the tool performs inference and returns a result WITHOUT persisting
  patient images by default; nothing is stored unless explicitly enabled with consent.
• Consent & retention — a planned account tier gates any storage behind explicit consent, with
  retention limits and one-tap deletion.
• Governance — an audit trail of predictions (not identities) supports quality monitoring and
  model-drift detection without exposing personal data.

ETHICAL REVIEW
• Safety-first framing — a fail-safe threshold minimises missed pneumonia; a false positive
  (healthy patient referred for review) is the safe direction, handled by the "confirm with a
  doctor" flow.
• Honesty about limits — the model is strongest on clear digital X-rays; performance on phone
  photos of adult films is stated openly, and borderline cases show "uncertain" rather than a
  confident call.
• Bias & fairness — training spans pediatric and adult populations, with per-population
  metrics reported; external validation on Sri Lankan data is on the roadmap.
• Human-in-the-loop — the system never replaces a clinician; it triages and explains, leaving
  the diagnosis to a qualified professional.
• Accessibility — bilingual (Sinhala/English) interfaces are planned to serve the communities
  that need it most.

# ======================================================================
REFERENCES (add at the end)

1. Rajpurkar P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays.
  arXiv:1711.05225.
2. Kermany D. et al. (2018). Identifying Medical Diagnoses by Image-Based Deep Learning.
  Cell 172(5).
3. He K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
4. Selvaraju R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
5. Rahman T. et al. (2021). COVID-19 Radiography Database. Kaggle / Computers in Biology and
  Medicine.
6. World Health Organization — Pneumonia fact sheets and global health statistics.

