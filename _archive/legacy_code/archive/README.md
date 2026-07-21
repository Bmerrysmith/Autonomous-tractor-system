# Historical code and notebooks

Everything under `archive/` is retained for provenance only. It is not an active import source,
supported training path, evaluator, inference runtime, or deployment component.

- `v1_baseline_retinanet/` through `v5_era/` preserve superseded experiments and known-invalid
  training/evaluation paths.
- `voc_era/` preserves the retired VOC conversion/splitting workflow.
- Notebook outputs are stripped from source control; executed evidence belongs in hashed external
  artifacts.
- Do not import archived modules from active code or copy their inference/treatment semantics.

The July 20 deep audit contains the detailed disposition and known defects for each historical
artifact.
