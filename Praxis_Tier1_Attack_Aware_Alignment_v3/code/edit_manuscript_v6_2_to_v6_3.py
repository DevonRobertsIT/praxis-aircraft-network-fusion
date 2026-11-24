"""Build Roberts_Praxis_Chapters_1_5_v6_3.docx from v6.2.

Chapter 1 and Section 5.1 revision only; no result changes.
  * Section 1.4: one-sentence thesis statement (<= 30 words) naming the deliverable.
  * Section 1.6: former RQ3 split into single-goal RQ3-RQ5, matching H3', H4 and H5.
  * Section 1.7: one sentence stating the RQ-to-hypothesis mapping.
  * Section 5.1: RQ3 discussion split so that RQ3, RQ4 and RQ5 each follow their restated hypothesis.
  * Abstract: contribution named with the thesis deliverable ("audited evaluation framework").
Numbers added to Section 5.1 are those already reported in Section 4.7.3 and NUMBERS_FINAL.json
(H3' kappa 1 mean 0.010; H4 kappa 1 mean 0.021, t-interval [-0.003, 0.045]).
"""
import copy, sys, datetime
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

SRC, OUT = sys.argv[1], sys.argv[2]
d = Document(SRC)
body = d.element.body
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
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

def set_single_run_text(p, new, label):
    rs = runs(p._p)
    assert len(rs) == 1, f'{label}: expected one run, found {len(rs)}'
    set_rtext(rs[0], new)
    log.append(f'{label}: text set')

def replace(p, old, new, label):
    hits = [r for r in runs(p._p) if old in rtext(r)]
    assert len(hits) == 1 and rtext(hits[0]).count(old) == 1, f'{label}: target not unique in one run'
    set_rtext(hits[0], rtext(hits[0]).replace(old, new))
    log.append(f'{label}: replaced {old[:50]!r}')

def strip_ids(el):
    for node in el.iter():
        for a in ('paraId', 'textId'):
            node.attrib.pop('{%s}%s' % (W14, a), None)
    for tag in ('w:bookmarkStart', 'w:bookmarkEnd', 'w:proofErr'):
        for b in list(el.iter(qn(tag))):
            b.getparent().remove(b)
    return el

def clone_after(template, anchor_el, text):
    el = strip_ids(copy.deepcopy(template._p))
    rs = runs(el)
    assert len(rs) == 1
    set_rtext(rs[0], text)
    anchor_el.addnext(el)
    return el

# ---------------------------------------------------------------- 1.4 Thesis statement
THESIS = ('An audited evaluation framework will measure what combining aircraft surveillance and network data adds to '
          'attack detection in controlled tests, without crediting information unavailable in practice.')
n = len(THESIS.split())
assert n <= 30 and THESIS.count('.') == 1 and not THESIS.split()[0].lower().endswith('ing'), n
thesis_p = find(lambda t: t.startswith('A reproducible point-event framework can make the value and limits'), '1.4 thesis')
set_single_run_text(thesis_p, THESIS, '1.4 thesis statement')
log.append(f'1.4 thesis: {n} words, one sentence')

# ---------------------------------------------------------------- 1.6 Research questions
RQ3 = ('RQ3. Does lag/relation weighting improve the trained relational graph system’s ranking over its unweighted '
       'ablation on the intensity target at coupling strengths of at least 0.5?')
RQ4 = ('RQ4. Does access to relations and flow-level evidence improve the trained relational graph system’s ranking '
       'over an event-only model on the intensity target at zero and full coupling?')
RQ5 = ('RQ5. At zero alignment jitter, under which tested coupling strengths does fusion improve ranking over the '
       'validation-selected unimodal model within the same estimator family?')
rq3_p = find(lambda t: t.startswith('RQ3. What do lag/relation weighting'), '1.6 RQ3')
set_single_run_text(rq3_p, RQ3, '1.6 RQ3 (H3′)')
rq4_el = clone_after(rq3_p, rq3_p._p, RQ4); log.append('1.6 RQ4 (H4) inserted')
clone_after(rq3_p, rq4_el, RQ5); log.append('1.6 RQ5 (H5) inserted')

# ---------------------------------------------------------------- 1.7 mapping sentence
h_intro = find(lambda t: t.startswith('The following hypotheses govern the final evaluation.'), '1.7 intro')
replace(h_intro, 'The following hypotheses govern the final evaluation.',
        'The following hypotheses govern the final evaluation. H1′ through H5 answer RQ1 through RQ5 in order.',
        '1.7 intro mapping sentence')

# ---------------------------------------------------------------- 5.1 discussion
old_rq3 = find(lambda t: t.startswith('RQ3 separates three questions.'), '5.1 RQ3 paragraph')
# Section 5.1 layout: restated H3′, restated H4, then the combined RQ3 paragraph
h4_51 = Paragraph(old_rq3._p.getprevious(), d)
h3_51 = Paragraph(h4_51._p.getprevious(), d)
assert h4_51.text.startswith('H4 (relational evidence).'), '5.1 layout: H4 restatement'
assert h3_51.text.startswith('H3′ (lag/relation weighting). At a tested coupling strength κ ≥ 0.5'), '5.1 layout: H3′ restatement'

NEW_RQ3 = ('RQ3 isolates lag/relation weighting. The κ = 1 full-versus-unweighted difference of 0.010 does not support '
           'H3′’s 0.05 ranking margin, and its seed-level interval contains zero, although the component latency '
           'requirement is met. The κ = 0 contrast is diagnostic rather than a qualifying test.')
clone_after(old_rq3, h3_51._p, NEW_RQ3)
log.append('5.1 RQ3 paragraph inserted after the restated H3′')

NEW_RQ4 = ('RQ4 compares the full graph with an event-only model that has neither relations nor flow-node evidence. '
           'H4 is supported at κ = 0, where the full graph’s advantage over the event-only model is 0.064 with interval '
           '[0.021, 0.108]; it is not supported at κ = 1, where the 0.021 difference has interval [−0.003, 0.045]. '
           'The supported contrast concerns relations together with flow-node evidence.')
set_single_run_text(old_rq3, NEW_RQ4, '5.1 RQ4 paragraph (former RQ3 paragraph)')

h5_51 = Paragraph(old_rq3._p.getnext(), d)
assert h5_51.text.startswith('H5 (fusion detection threshold).'), '5.1 H5 restatement position'
h5_disc = Paragraph(h5_51._p.getnext(), d)
assert h5_disc.text.startswith('H5 has no qualifying zero-jitter condition'), '5.1 H5 discussion position'
replace(h5_disc, 'H5 has no qualifying zero-jitter condition',
        'RQ5 concerns the tested coupling strengths at which fusion improves on a unimodal comparator. '
        'H5 has no qualifying zero-jitter condition', '5.1 RQ5 lead sentence')

# ---------------------------------------------------------------- Abstract: name the same deliverable
abstract = find(lambda t: t.startswith('Security detectors whose evaluations draw on information'), 'abstract')
replace(abstract, 'The contribution is an audited evaluation procedure with explicit evidentiary boundaries',
        'The contribution is an audited evaluation framework with explicit evidentiary boundaries', 'abstract contribution')
n_abs = len(abstract.text.split()); assert 200 <= n_abs <= 350
log.append(f'Abstract length: {n_abs} words')

# ---------------------------------------------------------------- Version strings
replace(find(lambda t: t.startswith('Chapters 1–5, Version 6.2'), 'cover'), 'Version 6.2', 'Version 6.3', 'cover')
replace(find(lambda t: 'Version 6.2 is a manuscript revision' in t, 'B.2'), 'Version 6.2 is', 'Version 6.3 is', 'B.2')
cp = d.core_properties
assert 'version 6.2;' in cp.subject
cp.subject = cp.subject.replace('version 6.2;', 'version 6.3;')
cp.revision = (cp.revision or 1) + 1
cp.modified = datetime.datetime(2026, 9, 29, 23, 55, 0)
log.append(f'core.subject: {cp.subject}')

d.save(OUT)
print('\n'.join(log))
