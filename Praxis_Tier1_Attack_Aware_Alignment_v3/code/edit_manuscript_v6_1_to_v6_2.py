"""Build Roberts_Praxis_Chapters_1_5_v6_2.docx from v6.1.

Front-matter and Chapter 1 revision only; no result changes.
  * Abstract and Section 1.1: reliable assessment of combined evidence stated as the immediate problem.
  * Section 1.3: rubric-format problem statement with journal sources and an explanatory paragraph.
  * References: Allix et al. (2016) and Arp et al. (2024) added in alphabetical order.
Every edit asserts that its target text is found exactly once.
"""
import copy, sys, datetime
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.opc.constants import RELATIONSHIP_TYPE as RT

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

def replace(p, old, new, label):
    hits = [r for r in runs(p._p) if old in rtext(r)]
    assert len(hits) == 1 and rtext(hits[0]).count(old) == 1, f'{label}: target not unique in one run'
    set_rtext(hits[0], rtext(hits[0]).replace(old, new))
    log.append(f'{label}: replaced {old[:60]!r}...')

def strip_ids(el):
    for node in el.iter():
        for a in ('paraId', 'textId'):
            node.attrib.pop('{%s}%s' % (W14, a), None)
    for tag in ('w:bookmarkStart', 'w:bookmarkEnd', 'w:proofErr'):
        for b in list(el.iter(qn(tag))):
            b.getparent().remove(b)
    return el

def set_italic(r, on):
    rpr = r.find(qn('w:rPr'))
    if rpr is None:
        rpr = OxmlElement('w:rPr'); r.insert(0, rpr)
    for old in rpr.findall(qn('w:i')):
        rpr.remove(old)
    if on:
        i = OxmlElement('w:i')
        # schema order: rStyle, rFonts, b, bCs, i ...; place after b/bCs if present, else after rFonts
        anchor = None
        for tag in ('w:bCs', 'w:b', 'w:rFonts', 'w:rStyle'):
            anchor = rpr.find(qn(tag))
            if anchor is not None:
                break
        if anchor is not None:
            anchor.addnext(i)
        else:
            rpr.insert(0, i)

# ---------------------------------------------------------------- Abstract
ABS_OLD = ('This praxis evaluates a framework for aligning sparse aircraft observations with network telemetry '
           'and interpreting the resulting model scores.')
ABS_NEW = ('Security detectors whose evaluations draw on information unavailable in practice can report gains that '
           'realistic testing does not sustain. This praxis examines that problem for combined aircraft–network evidence, '
           'evaluating a framework for aligning sparse aircraft observations with network telemetry and interpreting '
           'the resulting model scores.')
abstract = find(lambda t: t.startswith(ABS_OLD), 'abstract')
replace(abstract, ABS_OLD, ABS_NEW, 'Abstract opening')
n_abs = len(abstract.text.split())
assert 200 <= n_abs <= 350, n_abs
log.append(f'Abstract length: {n_abs} words (program range 200-350)')

# ---------------------------------------------------------------- Section 1.1
p123 = find(lambda t: t.startswith('Low-altitude aircraft observations provide a useful setting'), '1.1 para 2')
replace(p123, 'These findings motivate contextual analysis.',
        'These findings motivate examining surveillance data alongside other evidence.', '1.1 para 2 sentence 3')
replace(p123, 'or that unrelated network activity corroborates it.',
        'or that unrelated network activity corroborates it. Whether such a combination adds information must '
        'therefore be measured reliably rather than assumed.', '1.1 para 2 closing sentence')

p124 = find(lambda t: t.startswith('The engineering problem addressed here is how to construct'), '1.1 para 3')
replace(p124,
        'The engineering problem addressed here is how to construct and evaluate a joint analytical representation '
        'when the available aircraft and network records were collected independently.',
        'The immediate engineering problem is the reliable assessment of combined evidence: how to construct and '
        'evaluate a joint analytical representation from independently collected aircraft and network records so '
        'that a measured gain reflects the evidence rather than the evaluation design.',
        '1.1 para 3 sentence 1')

# ---------------------------------------------------------------- Section 1.3
PS = ('Of 30 reviewed security studies, 73% used some training information unavailable in practice '
      '(Arp et al., 2024); realistic testing revealed malware detection performance declines exceeding 88% '
      '(Allix et al., 2016).')
issue = 'Of 30 reviewed security studies, 73% used some training information unavailable in practice'
sowhat = 'realistic testing revealed malware detection performance declines exceeding 88%'
assert len(issue.split()) <= 20 and len(sowhat.split()) <= 10 and len(PS.split()) <= 30, 'problem statement limits'
log.append(f'Problem statement: issue {len(issue.split())} words, so-what {len(sowhat.split())} words, '
           f'total {len(PS.split())} words with citations')

EXPL_A = ('The two findings come from separate investigations. Arp et al. (2024) reviewed 30 papers published at the '
          'top four security conferences over ten years and found data snooping (training with data typically '
          'unavailable in practice) present in 57% of them; including papers in which it was partly present raises the '
          'share to 73%, the figure used in the problem statement. Allix et al. (2016) separately demonstrate the '
          'sensitivity of reported performance to evaluation conditions: their Android malware classifiers reached '
          'F-measures (a score balancing missed detections against false alarms) above 0.9 in laboratory validation '
          'but below 0.1 when tested ')
EXPL_I = 'in the wild'
EXPL_B = ('. The problem statement’s 88% is a relative decline calculated from those reported bounds, '
          '(0.9\u00a0−\u00a00.1)\u00a0÷\u00a00.9\u00a0≈\u00a00.89, and is therefore a minimum; it is not an average reported by the authors, an '
          'accuracy reduction, or evidence of the same effect in this praxis. Related evaluation risks arise when '
          'independently collected aircraft observations and network flows are combined for security analysis: '
          'replayed flows can recur across training and test partitions, and capture conditions can stand in for the '
          'label. Without distinguishing these effects, reported gains may not establish whether combining the sources '
          'adds useful information.')

old13 = find(lambda t: t.startswith('Sparse aircraft observations and independently captured network flows'), '1.3 body')
template = copy.deepcopy(old13._p)
(r0,) = runs(old13._p)
set_rtext(r0, PS)
log.append('1.3: previous paragraph replaced by the problem statement')

expl = strip_ids(template)
(t0,) = runs(expl)
t1 = copy.deepcopy(t0); t2 = copy.deepcopy(t0)
set_rtext(t0, EXPL_A); set_rtext(t1, EXPL_I); set_rtext(t2, EXPL_B)
set_italic(t1, True)
t0.addnext(t1); t1.addnext(t2)
old13._p.addnext(expl)
log.append('1.3: explanatory paragraph inserted after the problem statement')

# ---------------------------------------------------------------- References
cevik = find(lambda t: t.startswith('Çevik, N., & Akleylek, S. (2025)'), 'Cevik reference (template)')
akidau = find(lambda t: t.startswith('Akidau, T., Begoli, E.'), 'Akidau reference')
baltru = find(lambda t: t.startswith('Baltrušaitis, T.'), 'Baltrusaitis reference')
assert akidau._p.getnext() is baltru._p, 'Akidau must directly precede Baltrusaitis'

NEW_REFS = [
    ('Allix, K., Bissyandé, T. F., Jérome, Q., Klein, J., State, R., & Le Traon, Y. (2016). Empirical assessment of '
     'machine learning-based malware detectors for Android: Measuring the gap between in-the-lab and in-the-wild '
     'validation scenarios. ',
     'Empirical Software Engineering, 21', '(1), 183–211. ', 'https://doi.org/10.1007/s10664-014-9352-6'),
    ('Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., Cavallaro, L., & Rieck, K. '
     '(2024). Pitfalls in machine learning for computer security. ',
     'Communications of the ACM, 67', '(11), 104–112. ', 'https://doi.org/10.1145/3643456'),
]
anchor = akidau._p
for lead, ital, tail, url in NEW_REFS:
    el = strip_ids(copy.deepcopy(cevik._p))
    rs = runs(el)
    assert len(rs) == 4, 'template reference must have four runs'
    set_rtext(rs[0], lead); set_rtext(rs[1], ital); set_rtext(rs[2], tail); set_rtext(rs[3], url)
    (h,) = list(el.iter(qn('w:hyperlink')))
    h.set(qn('r:id'), d.part.relate_to(url, RT.HYPERLINK, is_external=True))
    anchor.addnext(el); anchor = el
    log.append(f'Reference added: {lead[:40]}...')

# ---------------------------------------------------------------- Version strings
replace(find(lambda t: t.startswith('Chapters 1–5, Version 6.1'), 'cover'), 'Version 6.1', 'Version 6.2', 'cover')
replace(find(lambda t: 'Version 6.1 is a manuscript revision' in t, 'B.2'), 'Version 6.1 is', 'Version 6.2 is', 'B.2')
cp = d.core_properties
assert 'version 6.1;' in cp.subject
cp.subject = cp.subject.replace('version 6.1;', 'version 6.2;')
cp.revision = (cp.revision or 1) + 1
cp.modified = datetime.datetime(2026, 9, 29, 23, 30, 0)
log.append(f'core.subject: {cp.subject}')

d.save(OUT)
print('\n'.join(log))
