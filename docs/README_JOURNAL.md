# Journal Paper Format

## Files

- **`conference_paper.tex`** - Now formatted as single-column journal paper (IEEE Transactions style)
- **`journal_paper.tex`** - Identical copy for clarity

## Format Changes Made

### Document Class
```latex
% OLD (Conference, two-column):
\documentclass[10pt,conference]{IEEEtran}

% NEW (Journal, single-column):
\documentclass[journal,12pt]{IEEEtran}
```

### Key Differences

1. **Single Column Layout**
   - Full page width for text
   - Better readability
   - More space for equations and algorithms

2. **Font Size**
   - 10pt → 12pt (journal standard)

3. **Author Block**
   - Conference style author blocks removed
   - Journal-style author with thanks/footnotes
   - Headers added with journal name

4. **Figure Widths**
   - Conference: `width=0.48\textwidth` (two-column)
   - Journal: `width=0.7\columnwidth` (single-column)
   - Figures are larger and more visible

5. **Headers/Footers**
   - Added `\markboth{}{}` for running headers
   - Journal name in header
   - Author names in footer

6. **Additional Packages**
   - `hyperref` for clickable links
   - `url` for better URL formatting
   - Colored links (blue) for references

## Journal Metadata

Current placeholders (update before submission):
- **Journal**: IEEE/ACM Transactions on Audio, Speech, and Language Processing
- **Volume/Issue**: Vol. XX, No. X
- **Date**: MONTH YEAR
- **Authors**: Anonymous Authors
- **Institution**: [To be added]
- **Email**: [To be added]

## Compilation

```bash
# Standard compilation
pdflatex journal_paper.tex
bibtex journal_paper
pdflatex journal_paper.tex
pdflatex journal_paper.tex

# Or using latexmk
latexmk -pdf journal_paper.tex

# Clean auxiliary files
latexmk -c
```

## Page Count

Journal format typically results in **fewer pages** than conference format due to:
- Single column (vs two-column)
- Larger font (12pt vs 10pt)
- More whitespace

Expected: **15-20 pages** in journal format (was ~10 pages in conference two-column)

## Submission Guidelines

Before submitting to IEEE Transactions, update:

1. **Author information**
   ```latex
   \author{John Doe, Jane Smith, et al.%
   \thanks{Manuscript received...}
   \thanks{The authors are with...}
   \thanks{Corresponding author: email@domain.com}}
   ```

2. **Journal header**
   ```latex
   \markboth{IEEE/ACM Trans. Audio Speech Lang. Process., Vol.~XX, No.~X, MONTH~YEAR}%
   {Doe \MakeLowercase{\textit{et al.}}: Blind RIR Estimation via Test-Time RL}
   ```

3. **Abstract footnote** (received/revised dates)

4. **Acknowledgments** section with funding sources

5. **Author photos and bios** (add at end before references)

## Advantages of Journal Format

✅ **More space** for detailed explanations  
✅ **Larger figures** (70% column width vs 48% in conference)  
✅ **Better equation layout** (no column breaks)  
✅ **Professional appearance** (journal standard)  
✅ **Easier reading** (single column flow)  
✅ **More references possible** (no strict page limit)  

## Content Unchanged

All technical content remains identical:
- 25+ equations with cross-references
- 4 figures with detailed analysis
- 7 tables (results, ablation, SOTA, architecture, hyperparameters)
- Algorithm pseudocode
- Complete appendix
- All sections and subsections

Only formatting changed to journal style.
