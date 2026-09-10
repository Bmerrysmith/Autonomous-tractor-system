# IEEE preparation — advisor checkpoint, 2026-09-09

This is a preparation note, not a manuscript or completed venue recommendation.
The [detector campaign](DETECTOR_CAMPAIGN_2026-09-09.md) and
[gate status](../GATE_STATUS.md) remain authoritative. The four existing paper
repositories remain the destinations for eventual manuscript sources.

## Proposed first contribution

Test whether one isolated change to classification supervision or head
normalization improves dense rice/weed detection under a fixed local compute
budget and explicit evaluation protocol. Document outcomes even if the measured
benefit is inconclusive or negative. A benefit over the matched WeedDet control
does not imply superiority to Faster R-CNN, novelty, or field generalization.

Current evidence supports a reproducibility and diagnostic narrative: checkpoint
geometry affected standalone evaluation; run identity and incomplete aggregation
needed repair; two alternatives pass a training-subset memorization gate that
the current incumbent fails. Comparative validation results remain pending.

## Simpler work queue

| Work item | Input already available | Completion criterion |
|---|---|---|
| Correct the existing paper outlines | Four repository READMEs and the evidence register | README corrections applied locally. Copied historical documents still require reconciliation against final experiment evidence. |
| Prepare the methods skeleton | Locked campaign and hashed manifests | Dataset, preprocessing, architecture, optimizer, BN policy, selection, terminal statistic, hardware, and exclusions each have a source pointer. Leave results blank. |
| Prepare figure specifications | Current model and experiment matrix | Architecture diagram; split/provenance flow; seed-level AP comparison; accuracy/latency comparison; class/size error examples. Generate numeric figures only from eligible finished runs. |
| Complete the 100-object human review | Local review packet, original image and annotation identities | Export all 100 judgments with reviewer identity; summarize class and box defects by stratum. |
| Prepare the limitations paragraph | Evidence register E01–E16 | Historical development split; unresolved capture groups; label audit scope; no unseen RiceSEG tiles for all-site initialization; single local GPU; no actuation/field results. |
| Build the advisor decision sheet | This note, final campaign results, current venue instructions | One lead paper, bounded claim, venue rationale, authorship/funding inputs, and concrete remaining evidence gaps. |

Methods and figures should point back to one authoritative experiment record. Do
not duplicate the same result as independent evidence across multiple papers.

## Corrections applied to existing paper outlines

The previous `weeddet-detector/README.md` treated failure to beat the stock
baseline as inherently unpublishable and prescribes migration as the program.
Those statements are too strong and conflict with the agreed bounded study.
Its freeze-57/58 and RiceSEG-incumbent descriptions are stale: the confirmed
screening incumbent uses ImageNet, freezes 48 pretrained backbone BN layers, and
retains nine random backbone BN layers plus head BN in training mode.

The previous `rice-detection-methodology/README.md` labeled the evidence ready
and the historical detector/reference comparison same-protocol. Initialization
and suppression differences must be reconciled first. Epoch oscillation alone
does not establish a threshold below which between-model effects are meaningless;
independent seed differences and their uncertainty are needed. Claims about
"most published numbers" require a defined literature sample and audit.

All four paper provenance files now separate the original snapshot source
`ed93be5` from reviewed campaign infrastructure `979627a`. The latter is a
development pin awaiting upstream review, not a research release or the source
of any new comparative result. Future experiment records must pin the revision
actually run; earlier diagnostics retain their local source archives over
`9304278`. Historical result provenance is not overwritten.

All four local READMEs now state the current scope and evidence limits. The
transfer outline retires the pooled-standard-deviation decision rule and stale
A100 estimates. The dataset outline corrects its claim that capture families
never cross splits and separates structural checks from label truth and rights.
Original README bytes, replacement bytes, hashes, and diffs are retained in
`../audit_artifacts/detector_campaign_2026-09-09/paper_outline_corrections/` relative
to the checkout. A separate `paper_source_pins/` artifact preserves before/after
bytes and hashes for the subsequent README/provenance reconciliation. These
eight document changes remain local and uncommitted in the four paper repositories.

## Three venue candidates for the advisor

These are preliminary fit assessments inferred from official scope statements,
not acceptance predictions. Recheck scope, article type, charges, and author
instructions after the contribution is settled; no fee or schedule is assumed.

| Candidate | Possible fit | Evidence needed before recommending it |
|---|---|---|
| [IEEE Access](https://ieeeaccess.ieee.org/) | Broad applied and multidisciplinary engineering research; initial candidate for a reproducible comparative detector study. | A clear contribution beyond debugging, sufficient comparisons, and documented limitations. Its open-access publication charge needs an identified funding source. |
| [IEEE Sensors Journal](https://ieee-sensors.org/ieee-sensors-journal/) | Official topics include sensor applications and machine learning/detection from sensor data. | Explain the sensing contribution and establish meaningful sensing/evaluation evidence; camera images alone do not establish novelty. |
| [IEEE Transactions on AgriFood Electronics](https://ieee-cas.org/publication/ieee-transactions-agrifood-electronics) | Precision agriculture and complete electronic systems in the agri-food chain. | Stronger fit if a concrete sensing/electronics system contribution is demonstrated. Current offline detector evidence may be too narrow for its systems emphasis. |

The simplest present direction is the bounded detector study, with Access as an
initial scope candidate. Do not expand into hardware or field integration solely
to fit a journal before the current study has an interpretable result.

## Publication preparation requirements

IEEE recommends checking a target journal's scope and author instructions before
submission, and permits submission to only one publication at a time. Use the
selected publication's template after that choice.
[Submission process](https://journals.ieeeauthorcenter.ieee.org/submit-your-article-for-peer-review/the-ieee-article-submission-process/),
[author tools and templates](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/).

Retain an AI assistance record for the eventual acknowledgments: IEEE requires
disclosure of AI-generated article content, identifying the system, affected
sections, and extent of use. Grammar/editing assistance is treated differently;
authors remain responsible for checking content and references.
[IEEE author policy](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/).

Record actual author contributions, affiliations/ORCIDs, funding, data/code
rights, and availability statements. Do not invent these or publish data whose
release rights remain unresolved. No manuscript has been submitted by this work.
