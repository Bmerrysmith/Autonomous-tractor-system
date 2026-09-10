# Detector analysis protocol — specified before confirmation

This specifies the uncertainty calculation required by the
[campaign](DETECTOR_CAMPAIGN_2026-09-09.md). No confirmation results exist yet.
Freeze this document's hash with the selected arm and matched control before
launching seeds 101–106. Changes after seeing outcomes require a dated amendment.

## Estimand and experimental unit

For each arm and seed, take the arithmetic mean of validation COCO AP at epochs
52, 56, and 60. These epochs describe one training trajectory; the experimental
unit is the seed, not the epoch, detection, annotation, or image. Best-checkpoint
AP remains a separate secondary statistic. Do not replace missing terminal epochs
with earlier/best epochs or average a different window for a failed seed.

Pair finalist and matched control by the six prespecified seeds. For seed s,
define d_s = terminal_AP(finalist, s) - terminal_AP(control, s). Report all six
differences, their mean, median, sample standard deviation, minimum, and maximum.
Do not remove poorly performing seeds as outliers.

This estimates variation across training seeds conditional on the fixed dataset,
protocol, source, and environment. It does not quantify farm/season variation or
resolve the historical split's uncertain capture independence. Seed pairing is
prespecified; using the same integer does not imply identical random operations
inside different architectures.

## Primary confidence interval

Use a two-sided paired Student-t interval for the mean difference:

    mean(d) +/- t_(0.975, 5) * sample_std(d, ddof=1) / sqrt(6)

The critical value is approximately 2.57058. The standard error uses six
seed-level differences; it never uses 18 terminal epochs as independent samples.
This is the standard paired-mean interval described by
[NIST](https://itl.nist.gov/div898/handbook/prc/section3/prc312.htm).

Its nominal coverage assumes independent, approximately normally distributed
seed differences. Six seeds provide little ability to diagnose departures from
that assumption. Publish the individual differences and make that limitation
explicit. Do not choose between t, bootstrap, or another method after comparing
which interval passes. A zero spread requires an artifact/precision audit before
interpreting a zero-width interval; identical rounded numbers are insufficient.

The campaign's accuracy benefit requires mean(d) >= 0.01 and the interval's lower
bound > 0. Its efficiency branch requires a measured reduction >= 20% in the
declared inference endpoint and the AP interval's lower bound > -0.005. Preserve
these exact thresholds. The intervals are nominal per comparison, not a blanket
95% assurance across all screening choices, metrics, or papers.

Declare the selected benefit rationale and the latency or memory endpoint before
confirmation. Report all predefined endpoints, including unfavorable ones. The
Faster R-CNN comparison is a contextual reference; passing against the matched
WeedDet control does not establish superiority to that reference. Additional
class/size metrics and best-checkpoint comparisons are descriptive secondary
analyses, not additional opportunities to select a successful primary result.

## Inference endpoint specification

Measure on the local RTX 4070, batch 1, without AMP/autocast, with each arm's
declared input size, and hard NMS 0.5 / score floor 0.05 / maxDets 100. Preserve
native proposal limits. Use each seed's epoch-60 checkpoint with `model.eval()`
and `torch.inference_mode()`. All tensors/models stay on the same device. GPU
synchronization surrounds each timed forward plus decode/postprocess; image-file
reading and preprocessing are timed separately and excluded from this endpoint.
This is GPU inference latency, not end-to-end robot latency.

Before screening accuracy is inspected, select 100 validation image IDs with
seed 20260909 and freeze their identities/hash. Use this same fixed ordered image
set for all runs. Perform 20 untimed warm-up images and five measured passes;
alternate finalist/control order by pass. Keep raw timings and report median and
p95 latency. Median latency is the gate endpoint; p95 is a secondary descriptor.
The sample has been prepared without prediction at
`../audit_artifacts/detector_campaign_2026-09-09/latency_sample.json` relative to
the checkout, SHA-256 `e47438440b5f30916a013df69cc0e3ec0928cdcb87682ca88d5c8a954029b684`.

Peak inference memory is maximum live CUDA allocation including model weights,
input, forward tensors, and postprocessing, after resetting peak counters before
the measured passes. Report reserved memory separately. Do not substitute the
resource pilots' training allocations. Use a fresh process per checkpoint; record
device/software metadata and unexpected concurrent GPU work.

For each seed, reduce timings to one median latency and one peak-memory value.
Report the six paired percentage reductions and their mean. The declared
efficiency endpoint's mean reduction must reach 20%; repetitions estimate timing
stability and are not extra training seeds. Preserve all repetitions. No timing
result has been generated by this document.

## Failure handling and final report

Aggregate only complete, checksum-verified runs with matching within-arm recipes,
exact epoch schedules, distinct planned seeds, and known initialization. Diagnostic
runs, unaudited resumes, and legacy artifacts lacking required evidence are
ineligible. Failed attempts still count against the GPU budget and remain listed.
Do not replace a failed seed with an easier seed or silently reduce the sample.

If the cap prevents completing the required design, report a budget feasibility
failure with completed descriptive evidence and a bounded revised proposal. If
the interval is too wide or the benefit threshold is missed, report inconclusive
or negative evidence as appropriate. Neither case authorizes an expanded search.
Final test evaluation occurs only after selection/configuration is frozen and
must retain the historical split limitations. Predeclare one final evaluation
batch containing the epoch-60 checkpoint from every completed confirmation arm
and seed; report those seed-level results without choosing a best test seed or
checkpoint. Its epoch-60 statistic is distinct from the validation terminal mean.
Charge evaluation runtime to the existing budget and make no test-guided changes.
