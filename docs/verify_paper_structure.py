#!/usr/bin/env python3
"""
Verification script for journal paper structure.
Checks figures, tables, equations, citations, and cross-references.
"""

import re
from pathlib import Path
from collections import defaultdict

def extract_labels(tex_file):
    """Extract all labels from the tex file."""
    labels = {'fig': [], 'tab': [], 'eq': [], 'sec': [], 'alg': []}
    
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Find all labels
    label_pattern = r'\\label{(\w+):([^}]+)}'
    matches = re.findall(label_pattern, content)
    
    for label_type, label_name in matches:
        if label_type in labels:
            labels[label_type].append(f"{label_type}:{label_name}")
    
    return labels

def extract_references(tex_file):
    """Extract all \\ref and \\eqref commands."""
    refs = defaultdict(list)
    
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Find all \ref{} and \eqref{}
    ref_pattern = r'\\(?:ref|eqref){(\w+):([^}]+)}'
    matches = re.findall(ref_pattern, content)
    
    for ref_type, ref_name in matches:
        refs[ref_type].append(f"{ref_type}:{ref_name}")
    
    return refs

def count_figures(tex_file):
    """Count figure environments."""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    fig_count = len(re.findall(r'\\begin{figure}', content))
    return fig_count

def count_tables(tex_file):
    """Count table environments."""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    tab_count = len(re.findall(r'\\begin{table}', content))
    return tab_count

def count_equations(tex_file):
    """Count equation environments."""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Count single equations and align environments
    eq_count = len(re.findall(r'\\begin{equation}', content))
    align_count = len(re.findall(r'\\begin{align}', content))
    
    return eq_count, align_count

def verify_citations(tex_file, bib_file):
    """Verify all citations exist in bibliography."""
    with open(tex_file, 'r') as f:
        tex_content = f.read()
    
    with open(bib_file, 'r') as f:
        bib_content = f.read()
    
    # Extract all \cite{} commands
    cite_pattern = r'\\cite{([^}]+)}'
    citations = re.findall(cite_pattern, tex_content)
    
    # Flatten multiple citations (e.g., \cite{a,b,c})
    all_cites = []
    for cite in citations:
        all_cites.extend([c.strip() for c in cite.split(',')])
    
    # Extract all @article, @inproceedings, etc. from bib
    bib_pattern = r'@\w+{([^,]+),'
    bib_keys = re.findall(bib_pattern, bib_content)
    
    # Check for missing citations
    missing = [c for c in all_cites if c not in bib_keys]
    unused = [b for b in bib_keys if b not in all_cites]
    
    return all_cites, bib_keys, missing, unused

def main():
    tex_file = Path('journal_paper.tex')
    bib_file = Path('references.bib')
    
    print("=" * 80)
    print("JOURNAL PAPER STRUCTURE VERIFICATION")
    print("=" * 80)
    print()
    
    # 1. Count elements
    print("1. DOCUMENT ELEMENTS")
    print("-" * 80)
    fig_count = count_figures(tex_file)
    tab_count = count_tables(tex_file)
    eq_count, align_count = count_equations(tex_file)
    
    print(f"   Figures: {fig_count}")
    print(f"   Tables: {tab_count}")
    print(f"   Equations (single): {eq_count}")
    print(f"   Equations (align): {align_count}")
    print(f"   Total equations: {eq_count + align_count}")
    print()
    
    # 2. Extract labels
    print("2. LABELS")
    print("-" * 80)
    labels = extract_labels(tex_file)
    
    for label_type, label_list in labels.items():
        if label_list:
            print(f"   {label_type}: {len(label_list)} labels")
            for label in sorted(label_list)[:5]:  # Show first 5
                print(f"      - {label}")
            if len(label_list) > 5:
                print(f"      ... and {len(label_list) - 5} more")
    print()
    
    # 3. Extract references
    print("3. CROSS-REFERENCES")
    print("-" * 80)
    refs = extract_references(tex_file)
    
    for ref_type, ref_list in refs.items():
        if ref_list:
            unique_refs = set(ref_list)
            print(f"   {ref_type}: {len(unique_refs)} unique references ({len(ref_list)} total)")
    print()
    
    # 4. Check for undefined references
    print("4. UNDEFINED REFERENCES CHECK")
    print("-" * 80)
    all_labels = set()
    for label_list in labels.values():
        all_labels.update(label_list)
    
    all_refs = set()
    for ref_list in refs.values():
        all_refs.update(ref_list)
    
    undefined = all_refs - all_labels
    if undefined:
        print(f"   ⚠ WARNING: {len(undefined)} undefined references found:")
        for ref in sorted(undefined)[:10]:
            print(f"      - {ref}")
    else:
        print("   ✓ All references are properly defined!")
    print()
    
    # 5. Check citations
    print("5. BIBLIOGRAPHY CHECK")
    print("-" * 80)
    if bib_file.exists():
        all_cites, bib_keys, missing, unused = verify_citations(tex_file, bib_file)
        
        print(f"   Citations in text: {len(set(all_cites))}")
        print(f"   Entries in .bib file: {len(bib_keys)}")
        
        if missing:
            print(f"   ⚠ WARNING: {len(set(missing))} citations not in .bib file:")
            for cite in sorted(set(missing))[:10]:
                print(f"      - {cite}")
        else:
            print("   ✓ All citations found in bibliography!")
        
        if unused:
            print(f"   ℹ INFO: {len(unused)} unused bibliography entries")
    else:
        print("   ⚠ references.bib not found!")
    print()
    
    # 6. Summary
    print("6. SUMMARY")
    print("-" * 80)
    print(f"   Total labeled figures: {len(labels['fig'])}")
    print(f"   Total labeled tables: {len(labels['tab'])}")
    print(f"   Total labeled equations: {len(labels['eq'])}")
    print(f"   Total sections: {len(labels['sec'])}")
    print(f"   Total algorithms: {len(labels['alg'])}")
    print()
    
    # Expected counts based on the paper
    expected_figs = 12
    expected_tabs = 13
    expected_eqs = 39
    
    print("   Expected vs. Found:")
    print(f"      Figures: {len(labels['fig'])} / {expected_figs} {'✓' if len(labels['fig']) >= expected_figs else '⚠'}")
    print(f"      Tables: {len(labels['tab'])} / {expected_tabs} {'✓' if len(labels['tab']) >= expected_tabs else '⚠'}")
    print(f"      Equations: {len(labels['eq'])} / {expected_eqs} {'✓' if len(labels['eq']) >= expected_eqs else '⚠'}")
    print()
    
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
