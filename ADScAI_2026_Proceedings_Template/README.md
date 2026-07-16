# ADScAI 2026 Author Template & Submission Guidelines

Welcome to the 2nd Annual Conference on Data Science and Artificial Intelligence (ADScAI 2026), Department of CSE, University of Moratuwa.

## Submission Format Requirements
* **Page Limit:** Full papers must not exceed 8 pages, including references.
* **Template:** Use the `adscai26.tex` file provided here. It is **modeled after ACM `sigconf` style**, but customized for ADScAI 2026. ACM rights/DOI blocks are removed.
* **Main File:** `adscai26.tex` is the root file. Sections are included via `\input{}` commands.
* **Anonymity:** ADScAI 2026 follows a **Double-Blind Review** process. Please ensure:
    * Author names and affiliations are replaced with placeholders.
    * Self-citations are handled in the third person.
    * Remove acknowledgments in the initial submission.
* **Reference Style:** Use the `ACM-Reference-Format.bst`. Ensure all references have complete metadata (Year, DOI/URL, etc.).
* **Figures:** Figures should be high-resolution (≥300 dpi) and use the `\begin{figure}` environment.

## Common Errors to Avoid
1. **ACM Rights/DOI Blocks:** Do not use the default ACM copyright text. Our template is pre-set to `\setcopyright{none}`.
2. **CCS Concepts:** Papers should include Computing Classification System (CCS) concepts. Guidance is provided; ACM tool usage is optional.
3. **Formatting Violations:** Do not manually adjust spacing or margins. Use the template as-is. The default font is Libertine.
4. **Reference Formatting:** Ensure all authors’ names are fully written (e.g., John Smith, not J. Smith).

## Files Included
* `adscai26.tex`: The main skeleton for your submission.
* `acmart.cls`: Official ACM document class (used as a base; copyright blocks removed).

## Submission Checklist
- [ ] Main file uses `adscai26.tex` as root
- [ ] Paper anonymized for double-blind review
- [ ] Figures high-resolution (≥300 dpi)
- [ ] References formatted correctly
- [ ] No copyright / DOI blocks included

*Template version:* 1.0 | *Contact:* adscai@cse.mrt.ac.lk
