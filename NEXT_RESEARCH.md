# Next Research Session Handoff

## 1. Purpose

This document transfers the relevant context from the previous session for a
new research project. The target venue is **NAACL Main Conference**; the
deadline is not currently a constraint.

The next session should read this document before proposing experiments. It
should treat the project as a distinct follow-up study, not as an extension
that merely adds a new visualization method to the existing EMNLP paper.

## 2. Existing Paper

### Publication

- **Title:** *Empathy Is Steerable but Multi-Axial: Mechanism Geometry and
  Persona Effects in LLMs*
- **Venue:** EMNLP 2026 Main Conference
- **Repository overview:** [`README.md`](README.md)
- **Reproducibility guide:** [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- **Final paper:**
  [`Empathy_Is_Steerable_But_Multi_Axial__Mechanism_Geometry_and_Persona_Effects_in_LLM_camera.pdf`](Empathy_Is_Steerable_But_Multi_Axial__Mechanism_Geometry_and_Persona_Effects_in_LLM_camera.pdf)

### Scope

The paper studies **EPITOME-defined supportive empathy in peer-support
dialogue**, not empathy universally. EPITOME divides supportive communication
into:

- **ER (Emotional Reactions):** emotional acknowledgment;
- **IP (Interpretations):** communicating an understanding of the seeker's
  situation; and
- **EX (Explorations):** inviting the seeker to elaborate.

### Models and method

- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct
- Contrastive Activation Addition (CAA)
- One candidate residual-stream direction per ER/IP/EX dimension
- Layer and intervention-strength sweeps
- Persona-conditioned generation and activation-shift analysis

### Main findings

1. Layer 15 is a consistent cross-model intervention point: positive ER, IP,
   and EX steering changes the corresponding EPITOME proxy score in the
   expected direction across all three model families.
2. The recovered directions are only partly separable. Steering one dimension
   also changes non-target dimensions.
3. Persona prompts substantially change EPITOME scores, but persona-induced ER
   shifts align weakly with the recovered ER direction.
4. The full ER/IP/EX subspace captures only **2.6--3.1%** of persona-mean
   squared activation-shift magnitude at layer 15.
5. The paper therefore distinguishes **target steerability** from
   **selective control**.

### Important limitations to address in follow-up work

- The primary outcomes are automated EPITOME proxy scores; human evaluation is
  still needed for perceived emotional understanding and support.
- The persona set is narrow and prompt-dependent.
- EPITOME IP is an externally defined communication behavior, not direct
  evidence of an internal emotion-understanding module.
- The existing paper localizes persona displacement relative to the ER/IP/EX
  basis but does not explain where the remaining persona effect arises.

### Follow-up questions explicitly left open in the paper

The final paper already identifies two concrete follow-up targets:

1. After removing the ER/IP/EX subspace, the persona residual remains
   structured and low-rank. The paper states that whether this structure
   encodes persona-specific style, social-category associations, or interactions
   with other attributes remains unknown (`latex/acl_latex.tex`, around the
   persona residual analysis).
2. The Limitations section states that the residual-PC analysis is descriptive
   and does not identify a causal persona mechanism. It also leaves the
   final-layer instability hypothesis unverified, explicitly mentioning a
   possible logit-lens analysis (`latex/acl_latex.tex`, Limitations).

These statements establish a legitimate research continuation, but they do
**not** by themselves eliminate incremental-publication concerns. The new paper
still needs a new object of explanation, new causal evidence, and a conclusion
that is not already contained in the EMNLP paper.

## 3. Initial Follow-up Idea

The initial idea was to present emotional-event narratives from datasets such
as Crowd-enVENT or support-seeking narratives from EPITOME, condition the model
on positive or negative personas, and inspect layer-wise changes using a logit
lens or sparse autoencoder (SAE).

This framing needs correction. The project must **not** claim to observe a
change in the model's own emotion. What can be studied is:

- the model's representation of another person's emotional situation;
- its inferred emotion and event appraisal;
- its response policy after forming that representation; and
- the causal relationship between those representations and its supportive
  response.

Likewise, `good persona` and `bad persona` are too vague. Use operationalized
conditions such as **compassionate/validating**, **cynical/dismissive**, and
**neutral observer**, with matched and paraphrased prompt variants.

## 4. Recommended Central Question

> **Which representational features account for persona-induced activation
> shifts outside the recovered ER/IP/EX span, and do those features alter
> emotional-event appraisal or only the downstream response policy?**

This directly follows the unexplained residual result while separating two
possible functions of the recovered features:

1. **Appraisal change:** the persona changes the model's inferred emotion,
   agency, responsibility, control, pleasantness, or related event appraisal.
2. **Response-policy change:** the event representation remains mostly stable,
   but the persona changes how the model communicates or responds.

A null result is also meaningful. If persona conditioning changes supportive
responses without materially changing event appraisal, it would indicate that
persona primarily acts on response policy rather than emotional understanding.

The shorter secondary formulation is:

> **Does persona conditioning change what the model infers about an emotional
> event, or only how it responds to that inference?**

## 5. Terminology Boundary

Do not conflate the following uses of `interpretation`:

- **Emotion/event appraisal:** the model's inference about how an experiencer
  evaluates and feels about an event. Crowd-enVENT provides labels relevant to
  this level.
- **EPITOME Interpretation (IP):** a communicative behavior in which a response
  expresses an understanding of the seeker's experience.

The relationship between these two levels is an empirical question. A strong
contribution would test whether persona features identified on EPITOME carry
appraisal information on Crowd-enVENT and, in the reverse direction, whether
Crowd-enVENT appraisal features predict or causally influence EPITOME behavior.

## 6. Proposed Study Structure

### Overall dataset priority

- **EPITOME should be the primary dataset** for identifying the unexplained
  persona residual because the 2.6--3.1% result was established on the EPITOME
  pipeline and response representations.
- **Crowd-enVENT should be a contrast and transfer dataset** for determining
  whether the recovered features reflect general event appraisal, affective
  content, or supportive-response policy.

Crowd-enVENT is related to emotion and appraisal, so it is not automatically a
matched `non-empathy control`. It can contribute evidence about domain and
construct specificity, but a claim that it fully resolves empathy specificity
would require a carefully matched multidimensional control design.

### Study A: Decompose the unexplained EPITOME persona residual

1. Reproduce the existing persona-conditioned response representations and the
   component outside `span(ER, IP, EX)`.
2. Encode the original persona and neutral activations separately with an SAE,
   then compute paired persona-induced differences in SAE latent space. Do not
   feed an isolated difference or orthogonal-residual vector directly into an
   SAE unless that use is separately validated; such vectors may be outside the
   SAE's training distribution.
3. Identify features that consistently distinguish **empathetic**,
   **cynical**, and neutral-person conditions. Decode their paired feature
   contributions and quantify how much of the persona displacement, especially
   its component outside the ER/IP/EX span, they reconstruct.
4. Compare each feature with the original ER/IP/EX directions and test whether
   it is aligned, orthogonal, hierarchically related, or model-specific.
5. Separate features associated with response style, social identity, topic,
   and supportive behavior using controlled input and output comparisons.

Using the existing empathetic/cynical contrast provides continuity and a paired
baseline, but reusing the prompt pair is not itself a contribution. Multiple
matched paraphrases and neutral controls are still required.

### Study B: Locate appraisal and response-policy divergence

1. Present Crowd-enVENT emotional-event descriptions without explicit emotion
   words where possible.
2. Use a fixed completion or classification context, for example:
   `The person most likely felt ...`.
3. Measure when emotion-category and appraisal information becomes decodable
   across layers.
4. Compare neutral, compassionate/validating, and cynical/dismissive persona
   conditions on the exact same event.
5. Evaluate on held-out writers and events to reduce memorization and
   writer-style leakage.
6. Compare hidden states at the end of the event description with states during
   response generation to determine whether persona effects arise during input
   appraisal or response planning.

The fixed prediction context is important. Projecting an arbitrary story-end
hidden state onto emotion words does not by itself provide a meaningful logit
lens measurement because the model is not necessarily predicting an emotion
word at that position. Claims about an earlier or delayed `emotion trajectory`
must therefore be supported by a fixed prediction context, calibrated probes or
a tuned lens, and token-by-layer statistics rather than selected trajectories.

### Study C: Appraisal-to-supportive-response transfer

1. Generate a supportive response for the same event or for an EPITOME seeker
   post under each persona condition.
2. Measure EPITOME ER/IP/EX outcomes and obtain human judgments on a meaningful
   subset.
3. Test whether layer-wise appraisal representations predict downstream ER/IP/
   EX behavior.
4. Test transfer in both directions while keeping discovery and evaluation
   splits explicit: whether EPITOME persona features encode Crowd-enVENT
   appraisal, and whether independently discovered appraisal features predict
   EPITOME ER/IP/EX behavior.

### Study D: Causal localization

Descriptive layer plots are insufficient for the main claim. At least one
causal intervention should be central:

- patch neutral story-end activations into persona-conditioned runs;
- patch persona-conditioned activations into neutral runs;
- ablate appraisal-associated SAE features;
- clamp, amplify, or suppress selected SAE features while preserving comparable
  reconstruction quality; or
- remove a persona-induced component at selected layers and token positions.

Measure whether the intervention changes both:

- forced emotion/appraisal predictions; and
- generated supportive behavior or human-rated emotional understanding.

Include random-feature, random-direction, and norm-matched controls.

## 7. Recommended Role of Each Interpretability Method

### Standard logit lens

Use only as an exploratory visualization or baseline. Directly applying the
final unembedding matrix to intermediate residual states can be brittle because
intermediate representations are not guaranteed to be aligned with the final
layer's basis.

It nevertheless has one precise supporting role: test the previous paper's
hypothesis that final-layer steering failures reflect tighter coupling between
the residual stream and next-token logits. This should be a bounded diagnostic,
not the main contribution or evidence of an internal emotional state.

### Tuned lens or supervised probes

Use for the primary layer-wise trajectory analysis. Relevant options are:

- a tuned lens with one affine translator per layer;
- linear probes for emotion categories and appraisal dimensions; and
- calibrated emotion verbalizers under a fixed prediction context.

Report selectivity controls, label permutations, held-out performance, and
probe complexity. Decodability alone does not establish causal use.

### Sparse autoencoders

Use SAEs as the primary candidate method for decomposing the unexplained
persona residual into sparse features. This matches the question `what
constitutes the remaining persona shift?` more directly than a vocabulary
projection. It is still not proof that each feature has a unique semantic
meaning. A feature should be called functionally relevant only after ablation,
activation, or patching changes the expected output.

The decomposition should operate on paired full activations: compare SAE
latent activations between persona and neutral conditions, decode selected
latent differences, and then measure their contribution to the original and
ER/IP/EX-orthogonal activation shifts. Direct SAE encoding of a mean-difference
or residual vector is not automatically valid.

Training and validating SAEs across three model families may make the project
too broad. A pragmatic design is:

- one primary open model for deep SAE and causal analysis; and
- one or two additional models for replication of the main behavioral and
  layer-wise findings.

If SAE checkpoint compatibility cannot be validated, drop the residual-feature
claim and instead center the paper on **tuned lens/probes plus activation
patching** for appraisal-versus-response localization. Do not retain a weak SAE
section merely to claim mechanistic interpretability.

Public checkpoints reduce training cost but introduce model-compatibility
trade-offs:

- **Llama Scope** provides SAEs for Llama-3.1-8B-Base, while the existing paper
  uses Llama-3.1-8B-Instruct. Transfer to the instruction-tuned model must be
  validated through reconstruction and downstream-behavior checks rather than
  assumed.
- **Gemma Scope** provides extensive public SAEs, including instruction-tuned
  coverage in parts of the release, but adopting Gemma changes the primary
  model family and weakens direct continuity with the EMNLP experiments.
- Training a new SAE on the exact instruction-tuned model provides the cleanest
  match but is the most expensive option.

PCA or UMAP may be used for visualization, but separation in a two-dimensional
plot is not evidence of a distinct mechanism or manifold.

## 8. Essential Controls

- Remove or mask explicit emotion labels and obvious emotion words.
- Use content-free persona baselines to estimate persona-only activation
  offsets.
- Match persona prompts for length, format, and instruction strength.
- Use several paraphrases for each persona condition.
- Separate persona effects from direct instructions to produce empathic or
  hostile language.
- Randomize event/persona pairing and use paired comparisons on identical
  inputs.
- Split Crowd-enVENT by writer, not only by row.
- Test multiple emotion verbalizers and multi-token label variants.
- Include label-permutation and random-probe controls.
- For SAEs, report reconstruction fidelity, sparsity, dead features, and
  downstream performance preservation.
- Use multiple generation seeds where decoding is stochastic.
- Include human evaluation for emotional understanding, support quality, and
  persona adherence on a pre-specified subset.
- Treat Crowd-enVENT personal-event text according to its license and privacy
  requirements; do not redistribute text without confirming permission.

## 9. Novelty Boundary with the EMNLP Paper

The follow-up paper must not merely repeat the existing pipeline with logit
lens or SAE plots. The following would be too incremental:

- reusing the same persona prompts and showing another projection metric;
- recovering sparse features that correlate with ER/IP/EX without causal
  validation;
- reproducing the finding that persona displacement lies outside the existing
  ER/IP/EX subspace; or
- claiming an internal empathy mechanism from automated output scores.

The new contribution should instead explain **where persona conditioning acts
between emotional-event comprehension and supportive response generation**.
The strongest result would identify and causally test a representation that
mediates persona-conditioned appraisal or response behavior and evaluate its
transfer between Crowd-enVENT and EPITOME.

The fact that the EMNLP Discussion and Limitations explicitly leave residual
decomposition and logit-lens verification open supports the follow-up
narrative. It does not make a paper novel if the result is only `the residual
contains emotion-related SAE features` or `late-layer logits change`.

Hou, Daumé, and Rudinger (NAACL 2025) provide an important behavioral
antecedent: they manipulate perceiver and experiencer social identities and
measure predicted emotion-intensity gaps. The follow-up project can build on
this motivation by asking where persona-dependent differences arise internally
and whether they are causal. Their existence means the paper should not claim
that persona-conditioned empathy differences are newly discovered.

## 10. Candidate Research Questions

- **RQ1:** Which SAE features account for persona-induced activation shifts
  outside the recovered ER/IP/EX span?
- **RQ2:** Do those features arise during emotional-event appraisal or primarily
  during supportive-response planning and generation?
- **RQ3:** Does intervening on the selected features causally change emotion
  inference, EPITOME behavior, or both?
- **RQ4:** Do persona-associated features identified on EPITOME transfer to
  Crowd-enVENT appraisal, and which components are dataset- or model-specific?

Do not include all four questions unless the experiments support a coherent
story. A focused paper centered on **RQ1--RQ3**, with RQ4 as external validity,
is preferable to a broad but descriptive benchmark.

## 11. Candidate Hypotheses

- A sparse subset of features explains a non-trivial share of the structured
  persona residual outside the ER/IP/EX span.
- Persona conditioning affects appraisal and response-policy representations
  differently across layers and token positions.
- Empathetic and cynical conditions may share a similar representation of the
  event while diverging more strongly during response generation.
- A subset of appraisal-associated features may predict or causally influence
  EPITOME IP/ER, but it should not be assumed that Crowd-enVENT appraisal and
  EPITOME IP occupy the same direction or subspace.
- Final-layer instability may correspond to larger and less selective changes
  in next-token logits, but this remains a secondary hypothesis until tested.

## 12. Evaluation Standard for NAACL Main

The topic fits NAACL areas including interpretability and analysis of NLP
models, sentiment/emotion analysis, semantics, dialogue, and human-centered
NLP. Topic fit alone is not sufficient. The project should provide:

- a precise non-anthropomorphic research question;
- a clear distinction from the EMNLP paper;
- held-out quantitative evidence rather than selected visual examples;
- causal intervention rather than decodability alone;
- robustness across prompt paraphrases and at least limited cross-model
  replication; and
- human validation for claims about perceived emotional understanding or
  supportive quality.

## 13. Open Decisions for the Next Session

1. Choose the primary paper core: residual-feature identification or
   appraisal-versus-response localization. The recommended synthesis uses the
   former as the object and the latter as its functional test.
2. Choose the primary model and SAE strategy: Llama Scope transfer, Gemma Scope
   with a new model family, or a newly trained instruction-model SAE.
3. Select the exact Crowd-enVENT emotion and appraisal labels to model.
4. Define persona conditions without directly instructing the desired output.
5. Decide whether Crowd-enVENT events will be converted into support-seeking
   prompts or used only for the appraisal stage.
6. Define the human-evaluation protocol and feasible sample size.
7. Select the principal causal intervention and pre-register its controls.
8. Determine which experiments can be reused as infrastructure without reusing
   results from the EMNLP paper.
9. Conduct a systematic related-work search before making any `first
   mechanistic account` or `little prior work` claim.

## 14. Recommended First Task in the New Session

The next session should begin by producing a one-page research specification
containing:

1. one central claim;
2. two or three research questions;
3. the exact input, persona, and output conditions;
4. the main causal test;
5. the primary and secondary evaluation metrics; and
6. an explicit statement distinguishing the project from the EMNLP paper.

Suggested opening prompt:

```text
Read NEXT_RESEARCH.md and the existing README.md first. Design a focused NAACL
Main follow-up study that explains the structured persona residual outside the
ER/IP/EX span. Do not treat an LLM as experiencing emotions, do not conflate
Crowd-enVENT appraisal with EPITOME IP, and do not propose a merely descriptive
logit-lens/SAE extension of the EMNLP paper. Start by defining one central
claim and the smallest experiment set that can causally test it.
```

## 15. External Feedback Synthesis

Three independent model reviews (Opus, DeepSeek, and Grok) were considered.
Their useful consensus is incorporated as follows:

- Replace `LLM emotion change` with operational claims about event appraisal,
  latent prediction, feature activation, and response policy.
- Use the empathetic/cynical contrast rather than undefined good/bad personas.
- Do not stop at logit-lens, SAE, PCA, or UMAP observations; perform causal
  ablation, amplification, steering, or activation patching.
- Use EPITOME as the continuity dataset and Crowd-enVENT as an appraisal
  contrast or transfer test.
- Analyze both layer and token position so input-side appraisal can be
  distinguished from generation-side policy.

The following suggestions are retained only as hypotheses or cautions:

- Claims that persona-conditioned temporal processing is largely unexplored
  require a systematic literature review.
- Layer depth and token position are computational axes, not direct evidence of
  psychological time or a model `feeling` an emotion.
- A standard logit lens cannot establish when emotion is `understood`; a tuned
  lens or controlled probe is needed for the main trajectory claim.
- SAE is well matched to residual decomposition, but feature labels and
  correlations are not causal explanations.
- Public SAEs are useful only if their model, layer, hook point, normalization,
  and instruction-tuning compatibility match or are validated.
- No `first mechanistic account` claim should be made solely from the currently
  identified related papers.

## 16. Primary Background References

- Crowd-enVENT: Troiano, Oberländer, and Klinger, *Dimensional Modeling of
  Emotions in Text with Appraisal Theories: Corpus Creation, Annotation
  Reliability, and Prediction*:
  <https://direct.mit.edu/coli/article/49/1/1/112909/>
- Tuned Lens: Belrose et al., *Eliciting Latent Predictions from Transformers
  with the Tuned Lens*: <https://arxiv.org/abs/2303.08112>
- Sparse autoencoders: Cunningham et al., *Sparse Autoencoders Find Highly
  Interpretable Features in Language Models*:
  <https://arxiv.org/abs/2309.08600>
- Behavioral antecedent: Hou, Daumé, and Rudinger, *Language Models Predict
  Empathy Gaps Between Social In-groups and Out-groups*:
  <https://aclanthology.org/2025.naacl-long.611/>
- Gemma Scope: Lieberum et al., *Gemma Scope: Open Sparse Autoencoders
  Everywhere All At Once on Gemma 2*:
  <https://arxiv.org/abs/2408.05147>
- Llama Scope: He et al., *Llama Scope: Extracting Millions of Features from
  Llama-3.1-8B with Sparse Autoencoders*:
  <https://arxiv.org/abs/2410.20526>
- NAACL topic scope: <https://2025.naacl.org/calls/papers/>
