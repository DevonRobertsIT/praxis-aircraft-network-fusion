"""Build Roberts_Praxis_Chapters_1_5_v6_1.docx from v6: add the seed-level assumption check to Section 4.7.3.
Numbers come from code/42_seed_level_assumption_checks.py (final_reproduction/seed_level_assumption_checks_v6_1.csv)."""
import copy, sys, datetime
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

SRC, OUT = sys.argv[1], sys.argv[2]
d = Document(SRC)
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'

def paras():
    return [Paragraph(el, d) for el in d.element.body.iterchildren() if el.tag == qn('w:p')]
def find(pred, label):
    hits = [p for p in paras() if pred(p.text)]
    assert len(hits) == 1, f'{label}: {len(hits)} matches'
    return hits[0]
def runs(p):
    return list(p._p.iter(qn('w:r')))
def rtext(r):
    return ''.join(t.text or '' for t in r.findall(qn('w:t')))
def set_rtext(r, s):
    ts = r.findall(qn('w:t'))
    for t in ts[1:]:
        r.remove(t)
    ts[0].text = s
    ts[0].set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
def replace(p, old, new, label):
    hits = [r for r in runs(p) if old in rtext(r)]
    assert len(hits) == 1, label
    set_rtext(hits[0], rtext(hits[0]).replace(old, new))
    print(f'{label}: {old!r} -> {new!r}')

SENTENCE = ('Shapiro–Wilk tests of the ten paired seed differences gave p > 0.05 for all eight contrasts in Table 4.15 (minimum 0.080), '
            'and exact sign-flip permutation tests over all 1,024 sign assignments reproduced the t-test results '
            '(H4 at κ = 0: 0.0098 versus 0.0085; H2′ at κ = 0 and 1: 0.023 and 0.031 versus 0.020 and 0.026), '
            'so no hypothesis decision changed.')
anchor = find(lambda t: t.startswith('The full graph exceeds the event-only model by 0.064'), 'H4 paragraph')
new = copy.deepcopy(anchor._p)
for a in ('paraId', 'textId'):
    new.attrib.pop('{%s}%s' % (W14, a), None)
rs = list(new.iter(qn('w:r')))
for r in rs[1:]:
    r.getparent().remove(r)
set_rtext(rs[0], SENTENCE)
anchor._p.addnext(new)
print('Section 4.7.3 assumption-check paragraph inserted after the H4 paragraph')

replace(find(lambda t: t.startswith('Chapters 1–5, Version 6'), 'cover'), 'Version 6', 'Version 6.1', 'cover')
replace(find(lambda t: 'Version 6 is a manuscript revision' in t, 'B.2'), 'Version 6 is', 'Version 6.1 is', 'B.2')
cp = d.core_properties
cp.subject = cp.subject.replace('version 6;', 'version 6.1;')
cp.revision = (cp.revision or 1) + 1
cp.modified = datetime.datetime(2026, 9, 29, 20, 0, 0)
print('core.subject:', cp.subject)
d.save(OUT)
