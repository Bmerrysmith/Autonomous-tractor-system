# Artifact inventory and cleanup record

This public-safe inventory records hashes and dispositions without publishing private Google Drive
IDs. Hashes are SHA-256 over the original container bytes.

| Artifact | Size (bytes) | SHA-256 | Disposition |
|---|---:|---|---|
| `AGRINAV_WORK_PACKAGE_2026-07-20.zip` | 66,579 | `7ffc464cec8208ab8f162e1b76e4d3aa4cc9a6af7da5bfc25f920fc359f77792` | Supplied recovery package; two Markdown documents preserved under `docs/audits/2026-07-20/`; original remains outside the repository |
| `archive/agrinav_github_FULL.zip` | 39,590 | `2e4db9fcac356e3a8b4735d7e6035ecfd22579fc0fb5e1bcd865636b89ad582d` | Removed from the working tree after confirming it is a redundant nested snapshot of the initial Git project |
| `data/rice_training_curated.zip` | 26,436,935 | `ba7ea4c8755bb8d1bb54ddb8532691bae085248fb28f584c86aee18392ad9849` | Removed after content comparison; 247 files matched the extracted local curation directory |
| `data/rice_training_curated_source.zip` | 26,436,935 | `3c9a6664b72ccd833e8863244d4a957872c6ebe3a5f802bbb31ec2036107edf3` | Removed as a second container with byte-identical uncompressed content |
| `rice_detection_coco_split.zip` | 116,354,807 | `c5a57ef79473801b2d4a7aa5d843f6d352f408c306fc3a1a6647d6f69586f6ec` | Historical detector split (audit-confirmed capture-series leakage, §see 2026-07-20 audit). Left outside the repository at `Claude\Projects\Autonomous driving tractor\rice_detection_coco_split.zip`; not yet moved to versioned artifact storage |
| `riceseg_annotation_set.zip` | 35,500,661 | `fe7b87822fa596d58acbf70f93f0871cef4d141aaaee1e5beaf9dea946898a64` | RiceSEG annotation export from the pre-recovery checkout; provenance and relationship to the current detector dataset (`docs/detector_dataset_card.md`) not yet reconciled. Left outside the repository at `Claude\Projects\Autonomous driving tractor\riceseg_annotation_set.zip` |

## Source-control baseline

- Recovered local source snapshot audited by the work package: `2a056ce`.
- Verified GitHub repository: `Bmerrysmith/Autonomous-tractor-system`.
- Verified default branch on 2026-07-20: `master`.
- The unrelated legacy `main` branch is not an authoritative project source and must not be merged
  into `master` without an explicit history review.

## Artifact policy

- Git contains source, small manifests, audit records, documentation, and tiny future test fixtures.
- Bulk datasets, checkpoints, executed notebook outputs, and release binaries belong in versioned
  artifact storage with hashes and access/licensing metadata.
- Private Drive IDs and personal-document locators are intentionally excluded from public docs.
- The historical T7d and RiceSEG run artifacts still need an immutable external release manifest;
  their availability is not implied by this repository.
