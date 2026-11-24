"""Build Roberts_Praxis_Chapters_1_5_v6.docx from v5.9.

Editorial and data-consistency corrections only; no change to any result.
Every edit asserts that its target text is found exactly where expected.
"""
import copy, re, sys, datetime
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC, OUT = sys.argv[1], sys.argv[2]
d = Document(SRC)
body = d.element.body
log = []

def paras():
    return [Paragraph(el, d) for el in body.iterchildren() if el.tag == qn('w:p')]

def find_para(pred, label):
    hits = [p for p in paras() if pred(p.text)]
    assert len(hits) == 1, f'{label}: expected 1 paragraph, found {len(hits)}'
    return hits[0]

def run_elems(p):
    # direct runs and runs inside hyperlinks, in document order
    return [r for r in p._p.iter(qn('w:r'))]

def run_text(r):
    return ''.join(t.text or '' for t in r.findall(qn('w:t')))

def set_run_text(r, text):
    ts = r.findall(qn('w:t'))
    for t in ts[1:]:
        r.remove(t)
    ts[0].text = text
    ts[0].set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

def replace_in_para(p, old, new, label):
    hits = [r for r in run_elems(p) if old in run_text(r)]
    assert len(hits) == 1 and run_text(hits[0]).count(old) == 1, f'{label}: "{old}" not uniquely in one run'
    set_run_text(hits[0], run_text(hits[0]).replace(old, new))
    log.append(f'{label}: "{old}" -> "{new}"')

def italicize_quoted(p, quoted, term, label):
    """Replace “term” (quoted) with term in italics; trailing punctuation inside the quotes moves outside."""
    hits = [r for r in run_elems(p) if quoted in run_text(r)]
    assert len(hits) == 1, f'{label}: quoted string not found in a single run'
    r = hits[0]
    before, after = run_text(r).split(quoted, 1)
    trailing = quoted[len('“') + len(term):-len('”')]  # e.g. '.' in “preregistered.”
    after = trailing + after
    parent = r.getparent(); idx = parent.index(r)
    mid = copy.deepcopy(r); tail = copy.deepcopy(r)
    set_run_text(r, before); set_run_text(mid, term); set_run_text(tail, after)
    rpr = mid.find(qn('w:rPr'))
    if rpr is None:
        rpr = OxmlElement('w:rPr'); mid.insert(0, rpr)
    for old_i in rpr.findall(qn('w:i')):
        rpr.remove(old_i)
    i_el = OxmlElement('w:i')
    # schema order: rStyle, rFonts, b, bCs, i ... insert after rFonts/b if present
    anchor = None
    for tag in ('w:bCs', 'w:b', 'w:rFonts', 'w:rStyle'):
        anchor = rpr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addnext(i_el)
    else:
        rpr.insert(0, i_el)
    parent.insert(idx + 1, mid); parent.insert(idx + 2, tail)
    log.append(f'{label}: {quoted} -> italic {term}')

def table_after_caption(prefix):
    els = list(body.iterchildren())
    for i, el in enumerate(els):
        if el.tag == qn('w:p') and Paragraph(el, d).style.name == 'Table Caption' and Paragraph(el, d).text.startswith(prefix):
            for el2 in els[i + 1:]:
                if el2.tag == qn('w:tbl'):
                    return Table(el2, d)
    raise AssertionError(f'table for {prefix} not found')

def set_cell(table, row_pred, col, old, new, label):
    rows = [r for r in table.rows if row_pred([c.text for c in r.cells])]
    assert len(rows) == 1, f'{label}: expected one row, found {len(rows)}'
    cell = rows[0].cells[col]
    assert cell.text.strip() == old, f'{label}: cell is "{cell.text}", expected "{old}"'
    runs = [r for p in cell.paragraphs for r in run_elems(p) if run_text(r)]
    joined = ''.join(run_text(r) for r in runs)
    assert joined.strip() == old
    set_run_text(runs[0], joined.replace(old, new)); [set_run_text(r, '') for r in runs[1:]]
    log.append(f'{label}: {old} -> {new}')


W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
def strip_ids(el):
    for a in ('paraId', 'textId'):
        el.attrib.pop('{%s}%s' % (W14, a), None)
    for b in list(el.iter(qn('w:bookmarkStart'))) + list(el.iter(qn('w:bookmarkEnd'))):
        b.getparent().remove(b)
    return el

def insert_after(anchor_el, new_el):
    anchor_el.addnext(new_el)
    return new_el

def clone_para_with_text(template_p, text):
    new = copy.deepcopy(template_p._p)
    runs = [r for r in new.iter(qn('w:r'))]
    # keep first run's formatting, drop the rest
    for r in runs[1:]:
        r.getparent().remove(r)
    for child in list(new):
        if child.tag in (qn('w:hyperlink'), qn('w:proofErr'), qn('w:bookmarkStart'), qn('w:bookmarkEnd')):
            new.remove(child)
    set_run_text(runs[0], text)
    return strip_ids(new)

# ---------------------------------------------------------------- 1. version strings
replace_in_para(find_para(lambda t: t.startswith('Chapters 1–5, Version 5.9'), 'cover'),
                'Version 5.9', 'Version 6', 'Cover version')
replace_in_para(find_para(lambda t: 'Version 5.8 is a manuscript revision' in t, 'B.2'),
                'Version 5.8 is a manuscript revision', 'Version 6 is a manuscript revision', 'B.2 version')

# ---------------------------------------------------------------- 2. numbers reconciled to frozen sources
t33 = table_after_caption('Table 3.3.')
row = lambda label: (lambda cells: cells[0].strip() == label)
set_cell(t33, row('S1 test (attack plateau)'), 4, '160,617', '176,506', 'Table 3.3 S1 test attack flows')
set_cell(t33, row('S1 test (attack plateau)'), 5, '16,494', '16,671', 'Table 3.3 S1 test attack-flow count')
set_cell(t33, row('S1 train (benign)'), 2, '25,308', '25,269', 'Table 3.3 S1 train span')
set_cell(t33, row('S1 train (benign)'), 4, '486,286', '615,728', 'Table 3.3 S1 train flows')
set_cell(t33, row('S1 validation (benign)'), 2, '4,523', '4,283', 'Table 3.3 S1 validation span')
set_cell(t33, row('S1 validation (benign)'), 4, '62,975', '116,394', 'Table 3.3 S1 validation flows')
set_cell(t33, row('S1 test (benign)'), 2, '4,523', '4,883', 'Table 3.3 S1 test benign span')
set_cell(t33, row('S1 test (benign)'), 4, '68,113', '134,627', 'Table 3.3 S1 test benign flows')
set_cell(t33, row('S2 train (attack plateau)'), 4, '801,152', '989,868', 'Table 3.3 S2 train flows')
set_cell(t33, row('S2 train (attack plateau)'), 5, '73,167', '89,547', 'Table 3.3 S2 train attack flows')
set_cell(t33, row('S2 validation (attack plateau)'), 4, '155,431', '201,360', 'Table 3.3 S2 validation flows')
set_cell(t33, row('S2 validation (attack plateau)'), 5, '13,987', '17,870', 'Table 3.3 S2 validation attack flows')

set_cell(table_after_caption('Table 4.4.'), row('Within-session stratified'), 4, '0.938', '0.952', 'Table 4.4 ROC-AUC')
set_cell(table_after_caption('Table 4.6.'), row('Mean'), 5, '−0.003', '−0.002', 'Table 4.6 mean full-ablation')
set_cell(table_after_caption('Table 4.12.'), lambda c: c[0].strip() == '120' and c[1].strip() == '1', 5,
         '−0.178 [−0.208, −0.121]', '−0.178 [−0.208, −0.120]', 'Table 4.12 seed range')
set_cell(table_after_caption('Table 4.14.'), lambda c: c[0].strip() == '1', 6, '0.950', '0.951', 'Table 4.14 late-fusion RF')
set_cell(table_after_caption('Table 4.16.'), lambda c: c[0].strip() == 'Regime, seed 42' and c[1].strip() == 'R-GCN (flow features)', 7,
         '0.43 / 0.31–0.36', '0.43 / 0.31–0.35', 'Table 4.16 Spearman range')
replace_in_para(find_para(lambda t: '0.047, 0.115, 0.137, and 0.108' in t, '3.10 widths'),
                '0.047, 0.115, 0.137, and 0.108', '0.047, 0.115, 0.136, and 0.108', 'Section 3.10 width')

# ---------------------------------------------------------------- 3. references
cai = find_para(lambda t: t.startswith('Cai, J., Xie, Y.'), 'Cai ref')
replace_in_para(cai, ', 94472–94492. ', '. ', 'Cai pages (unverifiable)')
replace_in_para(cai, 'https://doi.org/10.52202/085713-2841', 'https://openreview.net/forum?id=hJJnwcvE2M', 'Cai link text')
purton = find_para(lambda t: t.startswith('Purton, L.'), 'Purton ref')
old_purton = 'https://australasiantransportresearchforum.org.au/identification-of-ads-b-system-vulnerabilities-and-threats/'
new_purton = 'https://australasiantransportresearchforum.org.au/wp-content/uploads/2022/03/2010_Purton_Abbass_Alam.pdf'
replace_in_para(purton, old_purton, new_purton, 'Purton link text')
for rid, rel in d.part.rels.items():
    if rel.is_external and rel._target == 'https://doi.org/10.52202/085713-2841':
        rel._target = 'https://openreview.net/forum?id=hJJnwcvE2M'; log.append(f'rel {rid} -> OpenReview')
    if rel.is_external and rel._target == old_purton:
        rel._target = new_purton; log.append(f'rel {rid} -> ATRF PDF')

# ---------------------------------------------------------------- 4. quotation marks used for terms -> italics
italicize_quoted(find_para(lambda t: t.startswith('A target derived from varying attack intensity'), 'q1'), '“graded”', 'graded', '2.4.3')
p273 = find_para(lambda t: t.startswith('Protocol chronology determines'), 'q2')
italicize_quoted(p273, '“pre-specified”', 'pre-specified', '2.6.4 a')
italicize_quoted(p273, '“preregistered.”', 'preregistered', '2.6.4 b')
italicize_quoted(find_para(lambda t: t.startswith('where ϕ is the attack-flow fraction'), 'q3'), '“graded”', 'graded', '3.7')
italicize_quoted(find_para(lambda t: t.startswith('The margins are absolute average-precision differences'), 'q4'), '“practical margin”', 'practical margin', '3.10')
italicize_quoted(find_para(lambda t: t.startswith('Revised H1′–H3′ and their decision rules'), 'q5'), '“Pre-specified”', 'Pre-specified', '3.12')
italicize_quoted(find_para(lambda t: t.startswith('Table 4.1 identifies the event population'), 'q6'), '“Validated”', 'Validated', '4.2')

# ---------------------------------------------------------------- 5. acronyms at first use in the chapter
replace_in_para(find_para(lambda t: '11:49:36.907 UTC' in t, 'UTC'), '11:49:36.907 UTC', '11:49:36.907 Coordinated Universal Time (UTC)', 'Ch3 UTC')
replace_in_para(find_para(lambda t: t.startswith('One R-GCN normalization'), 'R-GCN'), 'One R-GCN normalization',
                'One relational graph convolutional network (R-GCN) normalization', 'Ch3 R-GCN')
replace_in_para(find_para(lambda t: 'Final intensity tables and XAI diagnostics' in t, 'XAI'), 'Final intensity tables and XAI diagnostics',
                'Final intensity tables and explainable artificial intelligence (XAI) diagnostics', 'Ch3 XAI')
note_tpl = find_para(lambda t: t.startswith('Note. The distinctions adapt event-representation'), 'note template')
for cap, text in [('Table 3.1.', 'Note. AUPRC = area under the precision–recall curve; NF3 = NF-UNSW-NB15-v3 NetFlow benchmark.'),
                  ('Table 3.5.', 'Note. CI = confidence interval.'),
                  ('Table 4.5.', 'Note. CI = confidence interval; DEFF = design effect; n_eff = effective event count.'),
                  ('Table 4.13.', 'Note. MLP = multilayer perceptron.')]:
    insert_after(table_after_caption(cap)._tbl, clone_para_with_text(note_tpl, text))
    log.append(f'{cap} note added: {text}')

# ---------------------------------------------------------------- 6. in-text references to uncited tables and Figure 4.5
for label, old, new in [
    ('Table 2.1', 'to match the analytical unit to observation density.', 'to match the analytical unit to observation density (Table 2.1).'),
    ('Table 2.2', 'A fusion claim therefore requires a demanding comparator.', 'A fusion claim therefore requires a demanding comparator (Table 2.2).'),
    ('Table 2.3', 'operationalizes the intended research question.', 'operationalizes the intended research question (Table 2.3).'),
    ('Table 2.5', 'address resampling within a fixed run.', 'address resampling within a fixed run (Table 2.5).'),
    ('Table 2.6', 'trust-enhancing for an operator.', 'trust-enhancing for an operator (Table 2.6).'),
    ('Figure 4.5', 'in the reported endpoint cells, but', 'in the reported endpoint cells (Figure 4.5), but')]:
    replace_in_para(find_para(lambda t, o=old: o in t, label), old, new, f'cite {label}')

# ---------------------------------------------------------------- 7. hypotheses restated where evaluated (verbatim from Section 1.7)
H = {k: find_para(lambda t, k=k: t.startswith(k), k) for k in
     ['H1′ (alignment fidelity).', 'H2′ (relational versus sequential).', 'H3′ (lag/relation weighting).',
      'H4 (relational evidence).', 'H5 (fusion detection threshold).']}
def restated(key):
    el = copy.deepcopy(H[key]._p)
    ind = el.find(qn('w:pPr')).find(qn('w:ind'))
    ind.set(qn('w:left'), '720')
    return strip_ids(el)
def after_heading(heading, keys):
    anchor = find_para(lambda t: t == heading, heading)._p
    for k in keys:
        anchor = insert_after(anchor, restated(k))
    log.append(f'restated {keys} after "{heading}"')
def before_para(start, keys):
    target = find_para(lambda t: t.startswith(start), start)._p
    for k in keys:
        target.addprevious(restated(k))
    log.append(f'restated {keys} before "{start[:40]}"')
after_heading('4.5 Timestamp Basis (H1′a)', ['H1′ (alignment fidelity).'])
after_heading('4.6.3 Fusion Against Within-Family Unimodal Comparators', ['H5 (fusion detection threshold).'])
after_heading('4.7.3 Ten-Seed Results on the Intensity Target (H2′, H3′, H4)',
              ['H2′ (relational versus sequential).', 'H3′ (lag/relation weighting).', 'H4 (relational evidence).'])
before_para('RQ1 concerns whether time semantics', ['H1′ (alignment fidelity).'])
before_para('RQ2 compares implemented relational', ['H2′ (relational versus sequential).'])
before_para('RQ3 separates three questions.', ['H3′ (lag/relation weighting).', 'H4 (relational evidence).'])
before_para('H5 has no qualifying zero-jitter condition', ['H5 (fusion detection threshold).'])

# ---------------------------------------------------------------- 8. Section 5.3 controlled-comparison contribution (numbered item 3)
S53 = ('The controlled-comparison contribution consists of two safeguards against misleading model comparisons. '
       'The first is the coupling-injection protocol, which introduces a known physical–cyber association at specified coupling '
       'strengths and alignment-jitter levels, with fixed partitions, labels frozen across jitter levels, and shuffled-anomaly '
       'controls (Sections 3.9 and 4.6). Because the association is known to be present, the protocol can locate a '
       'condition-specific detection threshold for each comparator and test whether attributions respond to the introduced '
       'association. A coupled condition that fails the rule therefore does not show that the sources are unrelated; the failure '
       'can instead reflect limited precision or a unimodal comparator that already captures the association. The second '
       'safeguard comes from the training-budget comparison: an insufficient full-batch schedule placed the relational model '
       'below the Transformer on the intensity target, an ordering that the validation-controlled schedule removed (Section '
       '4.7.2). Architecture comparisons should therefore be interpreted only after optimizer updates, validation behavior, and '
       'checkpoint selection confirm adequate training for each model. The introduced association is synthetic, so the protocol '
       'characterizes the evaluation procedure rather than natural coupling; another study can reuse both safeguards but must '
       'justify its own coupling form, strengths, and training budgets.')
ev = find_para(lambda t: t.startswith('The evaluation contribution is a reusable sequence of checks'), '5.3 evaluation')
insert_after(ev._p, clone_para_with_text(ev, S53)); log.append('5.3 controlled-comparison paragraph inserted as item 3')

# ---------------------------------------------------------------- 9. document properties
cp = d.core_properties
for attr in ('comments', 'subject', 'keywords', 'category', 'content_status'):
    v = getattr(cp, attr)
    if v and '5.8' in v:
        setattr(cp, attr, v.replace('version 5.8', 'version 6').replace('Version 5.8', 'Version 6')); log.append(f'core.{attr} updated')
cp.revision = (cp.revision or 1) + 1
cp.modified = datetime.datetime(2026, 9, 28, 23, 59, 0)

d.save(OUT)
print('\n'.join(log)); print(f'{len(log)} edits')
