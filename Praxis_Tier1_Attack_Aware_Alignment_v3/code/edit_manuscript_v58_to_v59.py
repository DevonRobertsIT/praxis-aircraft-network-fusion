import copy, re, unicodedata, docx
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
SRC='/mnt/user-data/uploads/Roberts_Praxis_Chapters_1_5_v5_8.docx'; DST='/mnt/user-data/outputs/Roberts_Praxis_Chapters_1_5_v5_9.docx'
d=docx.Document(SRC); P=d.paragraphs; log=[]
def find(pred, last=True):
    ix=[i for i,p in enumerate(d.paragraphs) if pred(p.text.strip())]
    assert ix, "paragraph not found"; return d.paragraphs[ix[-1] if last else ix[0]]
def append(p, sentence):
    r=copy.deepcopy(p.runs[-1]._r); p._p.append(r); run=p.runs[-1]; run.text=' '+sentence; log.append(sentence[:60])
def replace_text(p, old, new):
    t=p.text; assert old in t, old[:60]
    if len(p.runs)==1: p.runs[0].text=t.replace(old,new)
    else:
        for r in p.runs[1:]: r._r.getparent().remove(r._r)
        p.runs[0].text=t.replace(old,new)
# 1. abstract
ab=find(lambda s: s.startswith('This praxis evaluates a framework for aligning sparse aircraft'))
replace_text(ab, "Corrected relational graph models exceeded the Transformer by mean average-precision differences of 0.066 and 0.0297 at zero and full coupling across ten paired runs, but did not meet the relational-versus-sequential hypothesis rule. Additional weighting did not meet its 0.05 criterion.",
 "Every earlier graph-model comparison had trained the relational model for at most 50 gradient steps; once trained to convergence, it exceeded the Transformer by mean average-precision differences of 0.066 and 0.0297 at zero and full coupling across ten paired runs but did not meet the relational-versus-sequential hypothesis rule. At zero coupling, its relations and the flow-level evidence they carry added 0.064 (95% interval 0.021 to 0.108) over an event-only model, the one hypothesis supported under its stated rule. Lag and relation weighting did not meet its 0.05 criterion, and alignment error of 300 s reduced the best cyber model's average precision by 0.129, whereas the 120 s test did not meet its rule.")
# 2. Chapter 2 citations
append(find(lambda s: s.startswith('Some fusion studies combine measurements of a shared physical process')), "Multimodal learning more generally presumes that the modalities carry related information about a common phenomenon, which is what makes joint representation, alignment, and fusion worthwhile (Baltrušaitis et al., 2019); when modalities are collected independently, that premise does not hold by construction.")
append(find(lambda s: s.startswith('Geirhos et al. (2020) describe shortcut learning')), "Kapoor and Narayanan (2023) classify the same risk as leakage: features that act as proxies for the target, and duplicated or temporally overlapping records across training and test data, inflate apparent performance without reflecting the intended relationship.")
append(find(lambda s: s.startswith('A positive estimate and an improvement meeting a declared margin')), "Lakens (2017) formalizes the distinction: a smallest effect size of interest defines the practical margin, and equivalence tests evaluate that margin directly rather than inferring practical importance from statistical significance alone.")
append(find(lambda s: s.startswith('Protocol chronology determines whether a comparison was specified')), "Nosek et al. (2018) describe preregistration as a means of separating confirmatory from exploratory analysis and treat changes to a registered plan as acceptable when they are transparently reported, so that readers can judge which inferences were planned.")
p279=find(lambda s: s.startswith("Fidelity concerns the relationship between an explanation and the model"))
append(p279, "Samek et al. (2017) formalized this approach as region perturbation, in which evidence is removed in order of attributed relevance and the resulting decline in the model's output is compared across explanation methods.")
append(p279, "Hooker et al. (2019) caution that removing features without retraining can move inputs outside the training distribution, so a deletion effect may reflect distribution shift as well as reliance; their remove-and-retrain benchmark addresses this at substantial computational cost.")
append(find(lambda s: s.startswith('Probability sensitivity and ranking degradation measure different consequences')), "Output-based criteria such as the infidelity measure of Yeh et al. (2019), which compares attribution-weighted perturbations with the resulting change in the model's output, assess the first kind of consequence.")
append(find(lambda s: s.startswith('Stability describes an explanation')), "Yeh et al. (2019) formalize this property as explanation sensitivity, the largest change in an explanation under small input perturbations, and show that reducing sensitivity carefully can also improve fidelity.")
# 3. AI disclosure: B.3 and a front-matter statement
b3=find(lambda s: s.startswith('Generative AI assistance was used for manuscript drafting'))
B3=["Generative AI tools were used extensively in this praxis under the author's direction, as permitted by program guidance when attributed. Their contributions were as follows.",
 "OpenAI ChatGPT/Codex (September 2026) wrote and executed, at the author's direction, the source-recurrence audit; the leakage-safe version 2 re-alignment, the adaptation of the existing notebooks to it, and the pipeline rerun; the any-flow and matched control evaluations; the proxy ablations, paired random-forest bootstrap, and standardized latency scripts; and the independent validation script. It also drafted the two technical reports that document this work.",
 "Anthropic Claude (September 2026) independently verified the version 1–3 artifacts against the manuscript; identified the capture-session confound in the version 2 design and the insufficient training budget of the original R-GCN protocol; proposed and implemented the regime-stratified cross-regime alignment, the session-indicator and bin-cluster analyses, the revised H1′ test, the graded intensity target, the coupling-injection study, the training-budget ablation with its event-only control, and the training-only overlap normalization; wrote the corresponding alignment, analysis, aggregation, export, and reproduction scripts and the patches to notebooks 06 and 09; executed the alignment, tabular, and coupling analyses; drafted Chapters 1 and 3–5 and the front matter of manuscript versions 5 through 5.7; and suggested candidate references, which the author read before citing.",
 "Generative AI assistance was also used in later revisions for manuscript drafting and revision, citation reconciliation, document formatting, and code inspection.",
 "The author defined the research problem and data sources, executed all neural-network training and explanation notebooks on the author's workstation, decided which proposed designs and corrections to adopt, and reproduced the decisive checks. AI-generated suggestions are not treated as empirical evidence; numerical findings refer to the identified research artifacts. The author remains responsible for the accuracy of the manuscript, source attribution, interpretation of the code and results, and compliance with program and advisor requirements."]
b3.runs[0].text=B3[0]
for r in b3.runs[1:]: r._r.getparent().remove(r._r)
anchor=b3._p
for txt in B3[1:]:
    e=copy.deepcopy(b3._p); anchor.addnext(e); anchor=e; Paragraph(e, b3._parent).runs[0].text=txt
abh=find(lambda s: s=='Abstract', last=False)
h=copy.deepcopy(abh._p); abh._p.addprevious(h); hp=Paragraph(h, abh._parent)
for r in hp.runs[1:]: r._r.getparent().remove(r._r)
hp.runs[0].text='Use of Generative AI'
b=copy.deepcopy(ab._p); abh._p.addprevious(b); bp=Paragraph(b, abh._parent)
for r in bp.runs[1:]: r._r.getparent().remove(r._r)
bp.runs[0].text="Generative AI tools (OpenAI ChatGPT/Codex and Anthropic Claude) were used under the author's direction to write and execute analysis code, propose and implement corrected experimental designs, execute the alignment, tabular, and coupling analyses, and draft and revise manuscript text. Appendix B.3 details each tool's contribution and the author's role."
bp.runs[0].add_break(docx.enum.text.WD_BREAK.PAGE)
# 4. references
refs_new=[
 [("Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. ",0),("IEEE Transactions on Pattern Analysis and Machine Intelligence, 41",1),("(2), 423–443. https://doi.org/10.1109/TPAMI.2018.2798607",0)],
 [("Hooker, S., Erhan, D., Kindermans, P.-J., & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. Fox, & R. Garnett (Eds.), ",0),("Advances in Neural Information Processing Systems 32",1),(" (pp. 9737–9748). Curran Associates. https://proceedings.neurips.cc/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html",0)],
 [("Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. ",0),("Patterns, 4",1),("(9), Article 100804. https://doi.org/10.1016/j.patter.2023.100804",0)],
 [("Lakens, D. (2017). Equivalence tests: A practical primer for ",0),("t",1),(" tests, correlations, and meta-analyses. ",0),("Social Psychological and Personality Science, 8",1),("(4), 355–362. https://doi.org/10.1177/1948550617697177",0)],
 [("Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. ",0),("Proceedings of the National Academy of Sciences, 115",1),("(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114",0)],
 [("Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K.-R. (2017). Evaluating the visualization of what a deep neural network has learned. ",0),("IEEE Transactions on Neural Networks and Learning Systems, 28",1),("(11), 2660–2673. https://doi.org/10.1109/TNNLS.2016.2599820",0)],
 [("Yeh, C.-K., Hsieh, C.-Y., Suggala, A. S., Inouye, D. I., & Ravikumar, P. (2019). On the (in)fidelity and sensitivity of explanations. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. Fox, & R. Garnett (Eds.), ",0),("Advances in Neural Information Processing Systems 32",1),(" (pp. 10967–10978). Curran Associates. https://proceedings.neurips.cc/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html",0)]]
key=lambda s: unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode().lower()
def ref_paras():
    PP=d.paragraphs; r0=[i for i,p in enumerate(PP) if p.text.strip()=='References'][-1]; a0=[i for i,p in enumerate(PP) if p.text.strip().startswith('Appendix A')][-1]
    return [PP[i] for i in range(r0+1,a0) if PP[i].text.strip() and PP[i].style.name=='Reference']
tpl=ref_paras()[0]; rpr=tpl.runs[0]._r.find(qn('w:rPr'))
for segs in refs_new:
    text=''.join(s for s,_ in segs); rp=ref_paras(); nxt=next((p for p in rp if key(p.text)>key(text)), None)
    e=copy.deepcopy(tpl._p)
    for child in list(e):
        if child.tag in (qn('w:r'), qn('w:hyperlink')): e.remove(child)
    (nxt._p.addprevious(e) if nxt else rp[-1]._p.addnext(e)); np_=Paragraph(e, tpl._parent)
    for s,it in segs:
        r=np_.add_run(s)
        if rpr is not None: r._r.insert(0, copy.deepcopy(rpr))
        r.italic=bool(it)
# 5. version
v=find(lambda s: s.startswith('Chapters 1–5, Version 5.8')); replace_text(v,'Version 5.8','Version 5.9')
d.save(DST); print("saved; appended sentences:", len(log), "| references now:", len(ref_paras()))
