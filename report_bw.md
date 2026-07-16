---
title: "PneumoScan AI"
subtitle: "Brainstorm 2026 — Semifinal Report"
author: "Team GMora · University of Moratuwa"
---

# PneumoScan AI

**Explainable, fail-safe pneumonia screening from chest X-rays — built, validated, and deployed for Sri Lankan primary care.**

**Team GMora — University of Moratuwa**
Kusal Pabasara · Pasindu Mihiranga · Thanoj Buddhima · Senuja Dilmith

**Live application:** https://pneu.gmora.dev
**Theme:** Innovative and Affordable Healthcare Solutions for Local Challenges

---

## Executive Summary

Since the initial round, PneumoScan AI has progressed from a proof-of-concept model to a fully working, publicly deployed application — with a substantially stronger, independently validated model and a complete patient-and-clinician workflow.

| Metric | Previous | Now |
|---|---|---|
| Test accuracy | 86.7% | **94.2%** |
| AUC-ROC | 0.943 | **0.988** |
| Sensitivity | 96.7% | **94.3%** (fail-safe tuned) |
| Training images | 5,863 (pediatric) | **23,405 (pediatric + adult)** |

**What has been achieved**

- **A live, installable app** at https://pneu.gmora.dev — served over HTTPS, installable to a phone home screen (PWA), with in-browser camera capture and file upload.
- **A validated, generalised model** — retrained on a combined pediatric + adult corpus (23,405 images) with dedicated phone-photo robustness, verified by 5-fold cross-validation.
- **A fail-safe clinical design** — a sensitivity-first decision threshold, an image-quality gatekeeper, calibrated "uncertain" handling, and Grad-CAM explainability on every prediction.
- **Two audiences, one engine** — tuned plain-language guidance for patients/parents and a full technical read-out for clinicians.

> **Positive result, verified:** On a real phone photo of an adult pneumonia X-ray from a Moratuwa clinic, the system correctly returns "Signs of pneumonia"; on a healthy adult phone X-ray it returns "No clear signs" — the two cases the earlier pediatric-only model could not handle.

---

## 1. The Problem and Why It Matters

Pneumonia is a leading cause of mortality worldwide, and in Sri Lanka timely diagnosis is blocked by a critical shortage of radiologists — especially in rural and district facilities.

**Global and local burden**

- ~450 million pneumonia cases occur globally each year (WHO).
- ~2.5 million deaths annually — the leading infectious killer of children under five.
- Pneumonia accounts for roughly 15% of all deaths in children under five.
- In Sri Lanka, respiratory infections are among the top causes of hospital admission; rural centres face diagnostic delays of 24–72 hours awaiting radiologist review.
- Radiology is rarely available around the clock, leaving nights and weekends inadequately covered.

**Unmet clinical needs**

- **Radiologist shortage:** chest X-rays go unread or delayed in rural Sri Lanka where trained radiologists are scarce.
- **Diagnostic variability:** manual interpretation carries inter-observer variability, particularly for subtle or early pneumonia.
- **No affordable pre-screening:** no low-cost, AI-assisted triage tool is widely deployed in Sri Lankan hospitals.
- **24/7 unavailability:** specialist services are rarely continuous, leaving critical gaps in emergency care.

This directly aligns with the Brainstorm 2026 theme — *Innovative and Affordable Healthcare Solutions for Local Challenges.* An affordable, AI-assisted screen deployed in rural clinics and district hospitals can shorten dangerous diagnostic delays and prevent avoidable deaths.

**How we researched the problem**

- **WHO and global health data** to quantify the burden of pneumonia in South Asia.
- **Peer-reviewed clinical dataset** (Kermany et al., 2018; *Cell*) validating the authenticity of the pediatric training data.
- **Expert feedback** from the IEEE EMBS BioFusion Hackathon 2026, where domain experts reviewed the project and guided refinement of the model and app design.

---

## 2. The Solution — A Deployed Screening App

PneumoScan AI is a live web and mobile application. A health worker — or a patient — captures or uploads a chest X-ray and receives an instant, explainable, fail-safe screening result.

**Now live:** https://pneu.gmora.dev — HTTPS-secured, installable to a phone home screen, no app store required.

**How the app works — end to end**

1. **Capture or upload.** Take a photo of the X-ray with a phone camera, or upload a digital file (JPEG/PNG). A role selector tailors the experience for a patient/parent or a clinician.
2. **Quality gate.** An automatic check rejects unusable images (too small, no detail) and — for phone photos — tolerates the soft focus and warm colour cast that are normal for a snapshot of a film.
3. **Phone-photo rectification.** For photos of physical films, OpenCV detects the film boundary and corrects the camera-angle perspective.
4. **AI screening.** A fine-tuned ResNet-50 outputs a calibrated pneumonia probability; a fail-safe decision threshold converts it into a triage band.
5. **Explainable result.** A Grad-CAM heatmap shows which lung regions drove the prediction, alongside plain-language guidance and an explicit "screening aid, not a diagnosis" disclaimer.

**Result communication — safe by design**

| Band | Meaning | Guidance shown |
|---|---|---|
| No clear signs | Low pneumonia probability | Reassurance + "see a doctor if symptoms persist" |
| Uncertain | Borderline / on the fence | "Get a second opinion" — never a confident call |
| Signs of pneumonia | Above fail-safe threshold | Prompt to seek clinical care promptly |

**Two audiences, one engine**

- **For patients and parents:** plain-language result, clear next-step guidance, and a strong prompt to consult a clinician. No jargon.
- **For clinicians:** Grad-CAM overlay, class probabilities, the fail-safe operating point, and full technical detail for decision support.

---

## 3. Technical Implementation and Results

The model was substantially strengthened: broader training data, deeper fine-tuning, phone-photo robustness, and a fail-safe operating point — all validated on held-out data.

**Model and training strategy**

- **Backbone:** ResNet-50 (ImageNet-pretrained). We now fine-tune the deepest residual stage (layer4) plus the classifier head — approximately 15M trainable parameters — rather than the head alone, letting high-level features adapt from natural images to radiographs.
- **Combined corpus:** pediatric (Kermany/Guangzhou, 5,856) plus adult (COVID-19 Radiography Database, 17,549) = 23,405 images, near class-balanced (approximately 50/50).
- **Phone-photo augmentation:** training images are synthetically transformed to mimic phone snapshots of films — perspective, warm colour cast, glare, vignette, blur, JPEG — so the model reads real phone captures, not just clean digital scans.
- **Regularisation:** AdamW with weight decay, label smoothing, cosine learning-rate schedule, and early stopping — the train/validation gap stays approximately 1%.
- **Fail-safe threshold:** the decision threshold is tuned on a validation set for at least 95% sensitivity and capped so the tool errs toward flagging — a missed pneumonia is the costly error.

**Held-out test performance (3,510 images)**

| Metric | Value |
|---|---|
| Accuracy | 94.2% |
| Sensitivity (recall) | 94.3% |
| Specificity | 94.2% |
| Precision | 94.1% |
| F1-score | 0.942 |
| AUC-ROC | 0.988 |

**Improvement over the previous model**

| Metric | Old (pediatric-only) | Now (combined + phone-trained) |
|---|---|---|
| Test accuracy | 86.7% | **94.2%** |
| AUC-ROC | 0.943 | **0.988** |
| Sensitivity / Precision / F1 | 96.7% / 84.3% / 0.90 | **94.3% / 94.1% / 0.94** |
| Works on adult X-rays | No (fails) | **Yes — 92.4% recall, AUC 0.984** |
| Works on phone photos | No | **Yes — verified on real captures** |
| Overfitting (dev→test gap) | 7.7% | **1.1%** |

**Validation depth.** Beyond a single split, we ran 5-fold stratified cross-validation (sensitivity 98.7% ± 0.5% across folds) and report per-population results — pediatric (97.6% recall, AUC 0.995) and adult (92.4% recall, AUC 0.984) — demonstrating the model generalises across ages rather than overfitting one population.

### System Architecture and Live Deployment

A single trained model serves both audiences through a lightweight, low-cost, fully deployed web stack — evidence of functionality, not just simulation.

**Inference pipeline**

Capture / Upload → Quality gate (OpenCV) → Phone-photo rectification (perspective correction) → Preprocess (224×224, RGB, ImageNet normalisation) → ResNet-50 (fine-tuned) → Softmax P(Normal) / P(Pneumonia) → Fail-safe triage band → Grad-CAM heatmap → Role-aware result.

**Deployment stack (all live)**

| Layer | Technology | Role |
|---|---|---|
| Model | PyTorch 2.6, torchvision ResNet-50 | Trained on an NVIDIA RTX 3050 GPU; runs CPU-only inference in production. |
| Vision utilities | OpenCV, NumPy, Pillow | Quality checks, phone-photo rectification, Grad-CAM overlay. |
| App / UI | Streamlit + custom CSS / PWA | Patient/clinician UI, camera capture, installable home-screen app. |
| Server | AWS EC2 (Ubuntu), nginx, systemd | Reverse proxy, process supervision, always-on service. |
| Security | Let's Encrypt TLS (HTTPS) | Encrypted transport; required for browser camera access. |

**Delivered application features:** live at pneu.gmora.dev, installable PWA, in-browser camera capture, phone-photo rectification, image-quality gate, fail-safe threshold, calibrated "uncertain" band, Grad-CAM explainability, patient/clinician roles, clinical escalation guidance, offline app shell, and a model-insights dashboard.

**Feasibility demonstrated.** The entire system runs on free/low-cost infrastructure: open-source frameworks, a single small cloud VM for CPU inference, and no per-query licensing. An optional edge deployment (Raspberry Pi / Jetson Nano) enables fully offline use in clinics without internet.

---

## 4. Literature Review and Background

| Work | Contribution | Our advance |
|---|---|---|
| Rajpurkar et al. (2017) — CheXNet | 121-layer DenseNet, radiologist-level pneumonia detection. | We add explainability and a deployed, fail-safe workflow for low-resource use. |
| Kermany et al. (2018) | Transfer learning on pediatric CXR; expert-graded dataset. | We generalise beyond pediatrics to adults and phone photos. |
| Selvaraju et al. (2017) — Grad-CAM | Gradient-based visual explanation for CNNs. | Integrated into every prediction to build clinical trust. |
| He et al. (2016) — ResNet | Residual learning enabling deep networks. | Our fine-tuned backbone, adapted to CXR features. |
| Phillips et al. — CheXphoto | Photographed-CXR robustness benchmark. | We reproduce capture artefacts synthetically to train for phone use. |

**Gap addressed:** most published models are accurate on clean, in-distribution scans but are not deployment-ready, not explainable, and not validated on the messy inputs (adult anatomy, phone photos) that a real rural workflow produces. PneumoScan AI targets exactly that gap.

---

## 5. Feasibility

- **Technical feasibility.** Built entirely on mature, open-source frameworks (PyTorch, OpenCV, Streamlit). The trained model is approximately 90 MB and runs in well under a second on CPU — no GPU needed in production.
- **Operational feasibility.** Deployed on a single low-cost cloud VM with automated TLS and process supervision; an edge option removes the internet dependency for remote clinics.
- **Data feasibility.** Trained on 23,405 openly licensed, peer-reviewed images; the pipeline can ingest new regional data as partner hospitals contribute.
- **Economic feasibility.** Zero per-query licensing; runs on free-tier / low-cost infrastructure, making per-scan cost negligible versus enterprise tools.

---

## 6. Regulatory Aspects and Intellectual Property

As a healthcare tool, PneumoScan AI is positioned as Software as a Medical Device (SaMD) and framed explicitly as a screening decision-support aid, not a diagnostic device — keeping a clinician in the loop and lowering the regulatory risk class.

| Jurisdiction / framework | Applicability and our plan |
|---|---|
| NMRA (Sri Lanka) | Primary regulator. We target registration as a low–moderate risk SaMD, with a documented clinical-validation study on local X-rays before any clinical deployment. |
| US FDA | Reference framework: an equivalent product would pursue a 510(k) pathway as Class II. We follow FDA "Good Machine Learning Practice" and SaMD guidance for design controls. |
| Standards | Aligning with IEC 62304 (medical device software lifecycle) and ISO 14971 (risk management) as the engineering discipline for a future submission. |

**Intellectual property**

- The trained model, phone-photo augmentation pipeline, fail-safe threshold logic, and the patient/clinician application are our original contributions.
- All third-party components (ResNet-50 weights, datasets, libraries) are used under permissive open-source / open-access licences, documented for compliance.
- Strategy: retain proprietary rights over the application and locally-trained weights while keeping the core screening accessible to public hospitals (see sustainability model).

---

## 7. Data Security and Privacy Policy

- **In transit.** All traffic is encrypted over HTTPS (TLS). Camera access is only granted on a secure origin.
- **Data minimisation.** The current tool performs inference and returns a result without persisting patient images by default — nothing is stored unless explicitly enabled with consent.
- **Consent and retention.** A planned account tier gates any storage behind explicit consent, with retention limits and one-tap deletion.
- **Governance.** An audit trail of predictions (not identities) supports quality monitoring and model-drift detection without exposing personal data.

**Responsible-AI stance:** every result is labelled a screening aid requiring clinician confirmation; the tool is scoped and its known limitations (see ethics) are stated in-app.

---

## 8. Ethical Review

- **Safety-first framing.** A fail-safe threshold minimises missed pneumonia; a false positive (healthy patient referred for review) is the safe direction and is handled by the "confirm with a doctor" flow.
- **Honesty about limits.** The model is strongest on clear digital X-rays; performance on phone photos of adult films is stated openly, and borderline cases are shown as "uncertain" rather than a confident call.
- **Bias and fairness.** Training now spans pediatric and adult populations; per-population metrics are reported. Continued external validation on Sri Lankan data is part of the roadmap.
- **Human-in-the-loop.** The system never replaces a clinician — it triages and explains, leaving the diagnosis to a qualified professional.
- **Accessibility.** Bilingual (Sinhala/English) interfaces are planned so the tool serves the communities that need it most.

---

## 9. Impact on the Healthcare Sector

PneumoScan AI turns a 24–72 hour diagnostic delay into a sub-30-second, always-available assessment, directly addressing radiologist shortage, delay, and the absence of automated pre-screening.

| Stakeholder | Benefit |
|---|---|
| Rural health workers | Rapid AI-assisted pre-screening on a basic smartphone — faster decisions without specialist access. |
| Patients | Shorter waits, fewer missed cases, earlier treatment — improved outcomes. |
| District hospitals | A 24/7 second opinion that helps prioritise urgent cases and reduce radiologist load. |
| Ministry of Health | Scalable, low-cost deployment and a basis for national pneumonia-trend tracking. |

---

## 10. Sustainability, Cost, and Roadmap

- **Cost-effectiveness.** Open-source stack, CPU-only inference on a single low-cost VM, and zero per-query licensing make per-scan cost negligible — versus enterprise tools (Qure.ai, Lunit, Aidoc) priced out of reach for rural clinics.
- **Sustainability model.** Freemium access for NGOs and government hospitals; an optional subscription tier for private hospitals funds maintenance. The model improves as partner hospitals contribute regional data.

**Roadmap**

| Phase | Milestone |
|---|---|
| 1 · Done | Deployed binary screening with Grad-CAM, fail-safe threshold, adult + pediatric + phone-photo robustness. |
| 2 | Bacterial vs. viral pneumonia classification for targeted treatment guidance. |
| 3 | Additional local pathologies (tuberculosis, pleural effusion). |
| 4 | SaMD registration with NMRA Sri Lanka + local clinical-validation study. |
| 5 | Bilingual UI and integration with national health IT (eSuwa). |

**Social impact.** By making pneumonia screening fast, explainable, and affordable across rural Sri Lanka, PneumoScan AI can help prevent avoidable deaths — bringing specialist-level decision support to the settings that need it most.

---

## References

1. Rajpurkar P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays. arXiv:1711.05225.
2. Kermany D. et al. (2018). Identifying Medical Diagnoses by Image-Based Deep Learning. *Cell* 172(5).
3. He K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
4. Selvaraju R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
5. Rahman T. et al. (2021). COVID-19 Radiography Database. Kaggle / *Computers in Biology and Medicine*.
6. World Health Organization — Pneumonia fact sheets and global health statistics.
