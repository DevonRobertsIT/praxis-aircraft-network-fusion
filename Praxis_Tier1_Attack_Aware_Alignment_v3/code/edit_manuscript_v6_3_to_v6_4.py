"""Build Roberts_Praxis_Chapters_1_5_v6_4.docx from v6.3.

Section 1.4 only: the author's thesis statement replaces the v6.3 wording. No result changes.
"""
import sys, datetime
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

SRC, OUT = sys.argv[1], sys.argv[2]
d = Document(SRC)
body = d.element.body
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'
log = []

def paras():
    return [Paragraph(el, d) for el in body.iterchildren() if el.tag == qn('w:p')]

def find(pred, label):
    hits = [p for p in paras() if pred(p.text)]
    assert len(hits) == 1, f'{label}: expected 1 paragraph, found {len(hits)}'
    return hits[0]

def runs(p_el):
    return list(p_el.iter(qn('w:r')))

def rtext(r):
    return ''.join(t.text or '' for t in r.findall(qn('w:t')))

def set_rtext(r, s):
    ts = r.findall(qn('w:t'))
    for t in ts[1:]:
        r.remove(t)
    ts[0].text = s
    ts[0].set(XML_SPACE, 'preserve')

def replace(p, old, new, label):
    hits = [r for r in runs(p._p) if old in rtext(r)]
    assert len(hits) == 1 and rtext(hits[0]).count(old) == 1, f'{label}: target not unique in one run'
    set_rtext(hits[0], rtext(hits[0]).replace(old, new))
    log.append(f'{label}: replaced {old[:50]!r}')

OLD = ('An audited evaluation framework will measure what combining aircraft surveillance and network data adds to '
       'attack detection in controlled tests, without crediting information unavailable in practice.')
NEW = ('An audited point-event framework can assess the predictive value of combining independently collected aircraft '
       'and network records while identifying evaluation defects and defining limits on explainability claims under '
       'controlled conditions.')
heading = find(lambda t: t == '1.4 Thesis Statement', '1.4 heading')
thesis = find(lambda t: t == OLD, '1.4 thesis')
assert heading._p.getnext() is thesis._p, 'thesis must directly follow the 1.4 heading'
(r,) = runs(thesis._p)
set_rtext(r, NEW)
log.append(f'1.4 thesis statement replaced ({len(NEW.split())} words)')

replace(find(lambda t: t.startswith('Chapters 1–5, Version 6.3'), 'cover'), 'Version 6.3', 'Version 6.4', 'cover')
replace(find(lambda t: 'Version 6.3 is a manuscript revision' in t, 'B.2'), 'Version 6.3 is', 'Version 6.4 is', 'B.2')
cp = d.core_properties
assert 'version 6.3;' in cp.subject
cp.subject = cp.subject.replace('version 6.3;', 'version 6.4;')
cp.revision = (cp.revision or 1) + 1
cp.modified = datetime.datetime(2026, 9, 30, 0, 0, 0)
log.append(f'core.subject: {cp.subject}')

d.save(OUT)
print('\n'.join(log))
