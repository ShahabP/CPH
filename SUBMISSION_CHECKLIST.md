# Journal Submission Quick Checklist

**Paper:** Blind Room Impulse Response Estimation via Test-Time Reinforcement Learning  
**Date:** January 23, 2026  
**Status:** ✅ Ready for Submission

---

## Pre-Submission Verification

### Files Ready
- [x] `journal_paper.tex` - Main manuscript (97 KB, ~24-25 pages)
- [x] `figures/` - All 11 figures present (300 DPI PNG)
- [x] `algorithm_main.tex` - Algorithm pseudocode (optional)
- [x] `SUBMISSION_README.md` - Package documentation
- [x] All cross-references validated
- [x] No TODO/FIXME markers (except "TBD" for revision date)
- [x] Figure paths simplified (filenames only)

### Content Checklist
- [x] Abstract (~250 words)
- [x] Keywords (5 terms)
- [x] Introduction with clear contributions
- [x] Related work section
- [x] Method section with equations
- [x] Experimental setup description
- [x] Results with figures and tables
- [x] Discussion section
- [x] Conclusion
- [x] References (complete bibliography)

### Technical Validation
- [x] All 11 figures referenced in text
- [x] All 39 equations numbered and referenced
- [x] All 11 tables formatted with booktabs
- [x] No missing `\cite{}` or `\ref{}` commands
- [x] Consistent notation throughout
- [x] Proper mathematical formatting

---

## Compilation Test

**Commands:**
```bash
cd docs/
pdflatex journal_paper.tex
bibtex journal_paper
pdflatex journal_paper.tex
pdflatex journal_paper.tex
```

**Expected Output:**
- `journal_paper.pdf` (~24-25 pages)
- No compilation errors
- All figures embedded correctly
- All references resolved

**Verification Result:** ✅ Script passed all checks

---

## Submission Steps

### 1. Generate Final PDF
```bash
cd /Users/shahabpasha/Sonnet/Copenhagen/docs
pdflatex journal_paper.tex && bibtex journal_paper && pdflatex journal_paper.tex && pdflatex journal_paper.tex
```

### 2. Prepare Submission Package
**Required Files:**
- `journal_paper.pdf` (compiled PDF)
- `journal_paper.tex` (main LaTeX source)
- `figures/*.png` (all 11 figure files)
- `journal_paper.bbl` (compiled bibliography)

**Optional Files:**
- `algorithm_main.tex` (if requested)
- Source code ZIP (for reproducibility)

### 3. Online Submission
1. Go to journal submission portal
2. Create new submission
3. Upload PDF as main document
4. Upload LaTeX source + figures as supplementary
5. Fill in metadata:
   - **Title:** Blind Room Impulse Response Estimation via Test-Time Reinforcement Learning
   - **Abstract:** [Copy from paper]
   - **Keywords:** room impulse response, blind dereverberation, reinforcement learning, test-time optimization, acoustic signal processing
   - **Authors:** [De-anonymize and add affiliations]

### 4. Suggested Reviewers (Optional)
Consider experts in:
- Blind source separation
- Room acoustics and RIR estimation
- Reinforcement learning for signal processing
- Speech enhancement
- Audio deep learning

### 5. Cover Letter (Template)
```
Dear Editor,

We are pleased to submit our manuscript "Blind Room Impulse Response 
Estimation via Test-Time Reinforcement Learning" for consideration in 
IEEE/ACM Transactions on Audio, Speech, and Language Processing.

This work presents a novel approach to blind RIR estimation using 
test-time reinforcement learning, achieving 24.6-27.4 dB DRR across 
diverse acoustic conditions without requiring training data.

Key contributions:
1. Multi-head neural policy architecture
2. Composite acoustic reward function
3. Exponential decay initialization strategy
4. Training-free test-time optimization

We believe this work will be of significant interest to the TASLP 
readership given its novel approach to a fundamental audio processing 
problem and strong experimental validation.

All authors have approved the manuscript and agree with submission.

Sincerely,
[Author names and affiliations]
```

---

## Post-Submission

- [ ] Save manuscript ID from confirmation email
- [ ] Save submission date and time
- [ ] Track submission status online
- [ ] Prepare response plan for reviewer comments
- [ ] Update institutional repository (if required)

---

## Emergency Contact

**Verification Issues?**
Run: `python3 scripts/verify_submission.py`

**Figure Regeneration?**
Run: `python3 scripts/regenerate_figures.py`

**LaTeX Compilation Issues?**
Check: `docs/README_LATEX.md`

---

## Final Reminders

✅ **BEFORE CLICKING SUBMIT:**
1. Review PDF one final time
2. Check all author names and affiliations
3. Verify no typos in title/abstract
4. Confirm all figures are high quality
5. Double-check page limit compliance
6. Ensure no confidential information in anonymized version

✅ **REPRODUCIBILITY:**
- Code repository: ShahabP/Copenhagen
- All figures regeneratable: `python3 scripts/regenerate_figures.py`
- Full documentation in: `docs/SUBMISSION_README.md`

---

**Ready to Submit:** ✅ YES  
**Verification Status:** ✅ All checks passed  
**Last Updated:** January 23, 2026
