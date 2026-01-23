#!/usr/bin/env python3
"""
Verify journal paper submission package is ready.
Checks for common issues before submission.
"""

import re
from pathlib import Path


def check_tex_file():
    """Check LaTeX file for common issues."""
    tex_file = Path(__file__).parent.parent / "docs" / "journal_paper.tex"
    
    if not tex_file.exists():
        print("❌ journal_paper.tex not found!")
        return False
    
    content = tex_file.read_text()
    issues = []
    warnings = []
    
    # Check for TODOs/FIXMEs (excluding known TBD)
    todos = re.findall(r'(TODO|FIXME|XXX)(?!.*TBD)', content, re.IGNORECASE)
    if todos:
        issues.append(f"Found {len(todos)} TODO/FIXME markers")
    
    # Check for empty citations/references
    empty_refs = re.findall(r'\\(cite|ref|label)\{\s*\}', content)
    if empty_refs:
        issues.append(f"Found {len(empty_refs)} empty \\cite or \\ref commands")
    
    # Check figure paths (should be filenames only)
    bad_paths = re.findall(r'\\includegraphics.*\{[./]*/', content)
    if bad_paths:
        issues.append(f"Found {len(bad_paths)} figures with directory paths")
    
    # Check graphicspath is set correctly
    if r'\graphicspath{{figures/}}' not in content:
        warnings.append("graphicspath not set to {{figures/}}")
    
    # Count figures
    figures = re.findall(r'\\begin\{figure\}', content)
    print(f"✓ Found {len(figures)} figures")
    
    # Count equations
    equations = re.findall(r'\\begin\{equation\}', content)
    print(f"✓ Found {len(equations)} numbered equations")
    
    # Count tables
    tables = re.findall(r'\\begin\{table\}', content)
    print(f"✓ Found {len(tables)} tables")
    
    # Check abstract exists
    if r'\begin{abstract}' not in content:
        issues.append("No abstract found")
    else:
        print("✓ Abstract present")
    
    # Check keywords exist
    if r'\begin{IEEEkeywords}' not in content:
        warnings.append("No IEEEkeywords section found")
    else:
        print("✓ Keywords present")
    
    # Check document compiles
    if r'\begin{document}' not in content or r'\end{document}' not in content:
        issues.append("Document structure incomplete")
    else:
        print("✓ Document structure valid")
    
    # Print results
    print("\n" + "="*60)
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No critical issues found")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("="*60)
    
    return len(issues) == 0


def check_figures():
    """Check all figures exist and are referenced."""
    figures_dir = Path(__file__).parent.parent / "docs" / "figures"
    
    if not figures_dir.exists():
        print("❌ figures/ directory not found!")
        return False
    
    # List all PNG files
    figures = list(figures_dir.glob("*.png"))
    print(f"\n✓ Found {len(figures)} figure files:")
    
    expected = [
        "reward_components.png", 
        "training_convergence.png",
        "training_results_comprehensive.png",
        "drr_enhanced_results.png",
        "pesq_evaluation.png",
        "ablation_study.png",
        "method_evolution.png",
        "complexity_vs_performance.png",
        "spectral_domain_comparison.png",
        "room_dimensions_comparison.png",
        "neural_drr_vs_rt60_all_rir.png",
        "rir_evolution.png"
    ]
    
    missing = []
    for fig in expected:
        if not (figures_dir / fig).exists():
            missing.append(fig)
        else:
            size_kb = (figures_dir / fig).stat().st_size / 1024
            print(f"  ✓ {fig} ({size_kb:.0f} KB)")
    
    if missing:
        print("\n❌ Missing figures:")
        for fig in missing:
            print(f"  - {fig}")
        return False
    
    print(f"\n✅ All {len(expected)} required figures present")
    return True


def check_file_structure():
    """Check overall submission package structure."""
    base_dir = Path(__file__).parent.parent / "docs"
    
    required_files = [
        "journal_paper.tex",
        "SUBMISSION_README.md",
        "README_JOURNAL.md",
        "README_LATEX.md",
    ]
    
    print("\n" + "="*60)
    print("SUBMISSION PACKAGE STRUCTURE")
    print("="*60)
    
    all_present = True
    for file in required_files:
        if (base_dir / file).exists():
            print(f"✓ {file}")
        else:
            print(f"❌ {file} MISSING")
            all_present = False
    
    # Check figures directory
    if (base_dir / "figures").exists():
        print(f"✓ figures/ ({len(list((base_dir / 'figures').glob('*.png')))} files)")
    else:
        print("❌ figures/ MISSING")
        all_present = False
    
    return all_present


def main():
    """Run all verification checks."""
    print("="*60)
    print("JOURNAL SUBMISSION VERIFICATION")
    print("="*60)
    
    # Check file structure
    structure_ok = check_file_structure()
    
    # Check figures
    figures_ok = check_figures()
    
    # Check LaTeX file
    tex_ok = check_tex_file()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL VERIFICATION SUMMARY")
    print("="*60)
    
    if structure_ok and figures_ok and tex_ok:
        print("✅ SUBMISSION PACKAGE READY!")
        print("\nNext steps:")
        print("1. cd docs/")
        print("2. pdflatex journal_paper.tex")
        print("3. bibtex journal_paper")
        print("4. pdflatex journal_paper.tex (2x)")
        print("5. Review PDF for any formatting issues")
        print("6. Submit to journal portal")
        return 0
    else:
        print("❌ ISSUES FOUND - Please fix before submission")
        if not structure_ok:
            print("  - Fix file structure issues")
        if not figures_ok:
            print("  - Fix figure issues")
        if not tex_ok:
            print("  - Fix LaTeX issues")
        return 1


if __name__ == "__main__":
    exit(main())
