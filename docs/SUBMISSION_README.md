# Journal Paper Submission Package

## Paper Information
**Title:** Blind Room Impulse Response Estimation via Test-Time Reinforcement Learning

**Submission Date:** January 23, 2026

**Target Journal:** IEEE/ACM Transactions on Audio, Speech, and Language Processing

## Package Contents

### Main Submission Files

1. **journal_paper.tex** (97 KB)
   - Main LaTeX manuscript
   - ~24-25 pages, single-column format
   - 46 equations, 9 tables, 11 figures
   - Sections: Introduction, Related Work, Classical Baselines, Method, Implementation Details, Experimental Setup, Results, Discussion, Conclusion

2. **figures/** (11 publication-ready figures, 300 DPI)
   - `architecture_diagram.png` (211 KB) - Neural policy architecture
   - `reward_components.png` (220 KB) - Reward function components
   - `training_convergence.png` (386 KB) - Training convergence analysis
   - `drr_enhanced_results.png` (113 KB) - Main DRR performance results
   - `classical_baselines_comparison.png` (237 KB) - Baseline method comparison
   - `method_evolution.png` (213 KB) - Iterative improvement visualization
   - `complexity_vs_performance.png` (216 KB) - Computational trade-offs
   - `spectral_domain_comparison.png` (929 KB) - Frequency domain analysis
   - `room_dimensions_comparison.png` (146 KB) - Cross-room performance
   - `neural_drr_vs_rt60_all_rir.png` (174 KB) - RT60 robustness analysis
   - `rir_evolution.png` (175 KB) - RIR estimate evolution during training

3. **algorithm_main.tex** (413 B)
   - Standalone algorithm pseudocode
   - Can be included in main paper if required by reviewers

### Supporting Documentation

4. **README_JOURNAL.md** (3.3 KB)
   - Journal-specific formatting guidelines
   - Submission checklist

5. **README_LATEX.md** (4.6 KB)
   - LaTeX compilation instructions
   - Required packages and dependencies

## Compilation Instructions

### Prerequisites
- LaTeX distribution (TeX Live 2020+ or MiKTeX)
- Required packages (all standard):
  - geometry, cite, amsmath, algorithmic, algorithm
  - graphicx, booktabs, multirow, subcaption, hyperref

### Compile Commands
```bash
cd docs/
pdflatex journal_paper.tex
bibtex journal_paper
pdflatex journal_paper.tex
pdflatex journal_paper.tex
```

Or using latexmk:
```bash
cd docs/
latexmk -pdf journal_paper.tex
```

### Expected Output
- **journal_paper.pdf** (~24-25 pages)
- All figures should be embedded correctly from `figures/` subdirectory
- All cross-references and citations should resolve

## Figure Notes

All figures are:
- **Format:** PNG, 300 DPI (publication quality)
- **Size:** Optimized for single-column layout (width=0.8-0.9\textwidth)
- **Location:** `figures/` subdirectory (LaTeX uses `\graphicspath{{figures/}}`)
- **Naming:** Descriptive filenames matching content

## Paper Highlights

### Contributions
1. Multi-head neural policy architecture for RIR temporal structure
2. Composite acoustic reward function (6 components)
3. Exponential decay initialization strategy
4. Training-free test-time optimization

### Key Results
- **DRR Performance:** 24.6-27.4 dB across all conditions
- **Success Rate:** 100% achieving >7 dB target
- **Generalization:** Robust across RT60 100-1200 ms, volumes 39-5000 m³
- **Speed:** 2-3 minutes training per condition on standard GPU

### Comparisons
- 4 classical baseline methods (Spectral Subtraction, Wiener Filter, WPE, LMS)
- Comprehensive experimental validation
- Computational complexity analysis

## Reproducibility

### Code Repository
The complete codebase is available at:
- **Repository:** ShahabP/Copenhagen
- **Branch:** main
- **Key Scripts:**
  - `scripts/regenerate_figures.py` - Regenerate all paper figures
  - `scripts/baseline_comparison.py` - Classical method implementations
  - `scripts/spectral_comparison.py` - Spectral domain analysis
  - `src/neural_rir_agent.py` - Main RL agent implementation

### Regenerating Figures
All figures can be regenerated from scratch:
```bash
python scripts/regenerate_figures.py
```

This will create all 11 figures in `docs/figures/` with identical layout and quality.

## Pre-Submission Checklist

- [x] All figures placed in `figures/` subdirectory
- [x] LaTeX file uses `\graphicspath{{figures/}}` 
- [x] All `\includegraphics` commands use filenames only (no paths)
- [x] All cross-references verified (\ref, \cite, \label)
- [x] No TODOs or FIXMEs in manuscript (except "TBD" for revision date)
- [x] Abstract word count: ~250 words ✓
- [x] All equations numbered and referenced
- [x] All figures captioned and referenced in text
- [x] All tables formatted with booktabs
- [x] Author information anonymized for review
- [x] Manuscript compiled without errors
- [x] References formatted consistently

## Revision Notes

**Version:** 1.0 (Initial Submission)

**Major Revisions:**
- Added Section III: Classical Baseline Methods (3.5 pages, 15 equations)
- Added Section V: Implementation Details (4 pages, 9 equations)  
- Added 7 new figures (baselines + implementation)
- Regenerated 4 existing figures for improved clarity
- Total length increased from ~10 to ~24-25 pages

**Next Steps:**
1. Submit to journal submission portal
2. Upload PDF + LaTeX source + figures as separate files
3. Complete online submission form with abstract/keywords
4. Suggest 3-5 potential reviewers (optional)

## Contact Information

**Corresponding Author:** [To be added upon acceptance]

**Institution:** [To be added upon acceptance]

**Manuscript ID:** [Assigned by journal upon submission]

---

**Prepared by:** Anonymous Authors  
**Date:** January 23, 2026
