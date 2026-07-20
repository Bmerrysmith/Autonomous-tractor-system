# RiceSEG-derived curation set

This directory describes a local 245-image curation set selected from the upstream RiceSEG
semantic-segmentation dataset for possible future bounding-box annotation. It is not part of the
current detector training split.

## Public Git policy

Git versions only:

- `manifest.csv` — one row per selected tile with QA measurements;
- `AUDIT_REPORT.md` — the selection method and site-level counts;
- this provenance note.

The `aerial/` and `front/` image directories are intentionally ignored and excluded from the
public publish branch. Obtain the source dataset from its official distribution and confirm its
current access terms before reconstructing the local curation set.

## Upstream source and citation

- Dataset card: <https://huggingface.co/datasets/PheniX-Lab/RiceSEG>
- Paper: J. Zhou et al., “Global rice multiclass segmentation dataset (RiceSEG),” *Plant
  Phenomics*, 2025, DOI <https://doi.org/10.1016/j.plaphe.2025.100099>.
- The upstream dataset card labels RiceSEG as `mit` and requires users to accept gated-access
  conditions. The article itself is published under CC BY 4.0; that article license should not be
  assumed to replace the dataset's own terms.

No project-wide license has been selected for this repository. Do not infer redistribution rights
for local image copies from their presence on a developer machine.

## Local integrity snapshot (2026-07-20)

- 245 unique manifest rows and 245 image files: 95 aerial, 150 front;
- every manifest filename resolves exactly once;
- all images decoded during the recovery audit and were 512×512;
- two removed local ZIP containers had different container hashes but the same 247 non-directory
  entries and byte-identical uncompressed content; those entries also matched the extracted local
  directory byte-for-byte.

See `../../docs/ARTIFACT_INVENTORY.md` for the recorded archive hashes and disposition.
