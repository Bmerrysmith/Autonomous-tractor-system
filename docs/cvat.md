# CVAT — annotation and review

CVAT is **stage 4** of [`automated_annotation_pipeline.md`](automated_annotation_pipeline.md):
human review and adjudication, "the only truth step". Everything upstream of it
produces *proposals*; nothing becomes training truth without passing through
here. This page covers standing the stack up with Docker, loading the ontology
as labels, organizing data into projects and tasks, and getting reviewed
annotations back out.

Status: stack setup, the label generator and the CVAT-export converter are
implemented and tested. Docker Desktop is installed; no AgriNav data has been
imported into CVAT yet.

---

## 1. Install Docker

CVAT is distributed as a Docker Compose stack; there is no other supported way
to run it locally. Docker Desktop **is installed** on this machine as of
2026-08-31 (CLI at `%LOCALAPPDATA%\Programs\DockerDesktop`, Compose v5.4.0). Its
first launch stops at the license-agreement / WSL dialog, and the daemon does not
start until that is accepted — `npipe:////./pipe/dockerDesktopLinuxEngine: The
system cannot find the file specified` is what that looks like from the CLI.

1. If reinstalling: enable the **WSL2 backend**
   (Settings → General → "Use the WSL 2 based engine"), then enable your
   distro under Settings → Resources → WSL Integration.
   <https://docs.docker.com/desktop/install/windows-install/>
2. Confirm it works from this shell:

```bash
docker compose version
```

If the installer complains about an existing `docker-desktop` distro, let the
installer reuse or replace it. Removing it by hand
(`wsl --unregister docker-desktop`) **permanently deletes that distro's
contents** — only do that if the installer asks you to, and only after checking
`wsl -l -v` for anything you still want.

Sizing, measured on this machine 2026-08-21: 674 GB free on C:, and WSL sees 24
CPUs and ~15 GB RAM. CVAT images are roughly 8–10 GB, plus its database and cache
volumes, plus whatever you import. The stack is ~18 containers, so RAM is the
tighter limit of the two — cap WSL's memory in `%UserProfile%\.wslconfig` if
other work needs headroom. A full `pytest` run here can burn ~23 GB of temp on
its own; do not run both at once on a nearly-full drive.

## 2. Get CVAT

Clone it **outside** this repo — CVAT is a third-party application, not a
vendored dependency:

```bash
git clone https://github.com/cvat-ai/cvat ~/cvat
```

## 3. Apply the AgriNav overrides

Two files in [`deploy/cvat/`](../deploy/cvat/) configure the clone:

| File | Purpose |
| --- | --- |
| `docker-compose.override.yml` | publishes each dataset read-only under `/home/django/share` in every service that reads task files |
| `.env.example` | one path variable per dataset, plus `CVAT_HOST`, `CVAT_VERSION` |

Copy both into the CVAT clone, then edit the `.env`:

```bash
cp deploy/cvat/docker-compose.override.yml ~/cvat/docker-compose.override.yml
cp deploy/cvat/.env.example ~/cvat/.env
```

### Images are mounted, not uploaded

Nothing is copied and nothing is moved. Each dataset stays where it already
lives on disk and is bind-mounted read-only into the container, so a CVAT task
*references* the files. The alternative — dragging files through the CVAT web
UI — duplicates every byte into CVAT's database volume, which for these sets
means several GB of second copies and a long upload.

`read_only: true` means CVAT can read the pixels but cannot write into the
dataset it is annotating. Annotations live in CVAT's own database until you
export them.

### What is mounted

Counts measured 2026-08-21. Each row is one variable in `.env`.

| Mount point in CVAT | Dataset | Files | Size | Task |
| --- | --- | ---: | ---: | --- |
| `rice_phase2/train` | curated RICE, grouped split | 1,800 | 398 MB | detection |
| `rice_phase2/valid` | curated RICE, grouped split | 518 | 247 MB | detection |
| `riceseg` | RiceSEG expert masks (~3,078 rgb+label pairs) | 6,156 | 231 MB | segmentation |
| `rice_plant_masks` | "Rice Plant Image Dataset" (the misnamed RAR) | 5,391 | 1,557 MB | segmentation |
| `weed_species_v3` | BD weed V3, 11 species folders | 4,368 | 261 MB | classification |

**18,233 files, 2.69 GB**, each dataset present exactly once. That is the number
`find /home/django/share -type f | wc -l` should return once the stack is up.

### What is not mounted, and why

The directories these come from also hold ~17 GB of copies, supersessions, and
one unusable set. Mounting a parent directory would publish all of it, so the
override binds each dataset individually. Left out on purpose:

| Left out | Files | Reason |
| --- | ---: | --- |
| `rice_phase2/test` | 261 | **Sealed test split.** Unreachable from CVAT is what keeps it sealed. Adding the mount is the thing that un-seals it. |
| `extracted/rice_coco_annotated` | 2,579 | The same images as `rice_phase2`, in the native Roboflow folder split — the arrangement that put 231 of 261 intended test images into train/valid and voided two runs. |
| `extracted/rice_detection_coco` | 1,347 | Exact subset of the RICE set, verified by content SHA-256 at intake. |
| `extracted/weed_v4` | 3,632 | Exact subset of weed V3, same verification. |
| `extracted/weeddataset` | 1,761 | Git-LFS pointer stubs (~132 B each), no image content. Quarantined at intake. |
| `Downloads/RiceSEG/` | 6,156 | Second copy of the mounted RiceSEG set — same file count; `agrinav_data/incoming/extracted/riceseg` is the one mounted. |
| `agrinav_clean_annotated_2026-07-21/`, `agrinav_intake_2026-07-21/` | ~103k | Staging and deliverable trees (3.8 GB and 11.3 GB); every real image in them is already mounted above, once. |

The subset and stub findings are the intake audit's, recorded in that intake's
`_reports/`; they were established by content hash, not by filename.

## 4. Start it

```bash
docker compose up -d
```

Run that **in the CVAT clone** — compose reads `docker-compose.yml`,
`docker-compose.override.yml`, and `.env` from the working directory. First run
pulls several GB.

Create a login, then open <http://localhost:8080>:

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Verify the mounts landed before creating any task — a task built against a
missing share fails at import time, not at creation time:

```bash
docker compose exec cvat_server ls /home/django/share
```

Expect exactly four entries — `rice_phase2` (holding `train` and `valid`),
`riceseg`, `rice_plant_masks`, `weed_species_v3`. A missing one means a wrong
path in `.env`; an *extra* one
means something got mounted that the table in §3 says should not be. Count what
CVAT can actually see:

```bash
docker compose exec cvat_server find /home/django/share -type f | wc -l
```

18,233 on the mount set above, as measured 2026-08-21.

Stop with `docker compose down` (keeps volumes and therefore all annotation
work). `docker compose down -v` **deletes the database and every annotation in
it** — do not use it to "restart cleanly" unless you have exported first.

## 5. Load the labels from the ontology

Do not type labels into the web UI. `data/ontology.v1.json` is the canonical
class definition; a hand-typed CVAT project is an unversioned second copy that
drifts. Generate the spec instead:

```bash
agrinav data-cvat-labels --out artifacts/cvat/labels_polygon.json
```

Then in CVAT: **Projects → + → Raw** tab → paste the file's contents → Submit.
Create the project first and the tasks inside it, so every task inherits one
label set.

Variants:

```bash
agrinav data-cvat-labels --geometry mask --out artifacts/cvat/labels_mask.json
agrinav data-cvat-labels --geometry rectangle --allow-non-canonical-geometry \
  --out artifacts/cvat/labels_boxqa.json
```

`--geometry rectangle` requires the override flag on purpose. No canonical label
lists a box geometry, because the ontology principle is "boxes may be derived
from masks, but boxes are not canonical treatment geometry". A box project is a
proposal or QA layer; the flag makes that an explicit decision.

What the generator carries across, and why it is not a copy-paste job:

- **`treatment_eligible` and `verified_empty` become three-valued selects**, not
  checkboxes. Both are `boolean | null` in the ontology. A CVAT checkbox has no
  third state, so an untouched attribute would export as `false` — which is the
  forbidden inference "model returned no boxes ⇒ verified_empty". They default
  to `unknown`.
- **`annotation_confidence` has no default.** The ontology declares none, and
  inventing one records a confidence the annotator never stated.
- **Every attribute has a slot in the wire format.** `human_edit_action` is the
  per-shape accept / edit / delete decision, with the vocabulary from
  `data/schemas/annotation_record.v1.schema.json`; `review_status` appears only
  at image level, because the schema puts it on the record and sets
  `additionalProperties: false`, so a per-shape copy would have nowhere to land
  on export. A test fails if the CVAT spec and that schema ever drift apart.
- Image-level attributes hang off a `tag` label (`image_review`), since CVAT has
  no other place to put them. `review_status` there defaults to `unreviewed`.

Regenerate and re-paste whenever `data/ontology.v1.json` changes. CVAT keeps
existing annotations when you add labels or attributes; deleting a label deletes
its annotations.

## 6. Organize the data

**Project = one annotation contract.** One geometry, one label set, one
definition of done. The five mounts are three different tasks, so they are three
different projects — do not put a polygon review and a species classification in
the same one:

| Project | Geometry | Labels from | Mounts it draws on |
| --- | --- | --- | --- |
| `rice-detection-polygons` | polygon | `labels_polygon.json` | `rice_phase2/train`, `rice_phase2/valid` |
| `rice-segmentation` | mask | `labels_mask.json` | `riceseg`, `rice_plant_masks` |
| `weed-species` | tag | *(species vocabulary, not the detection ontology — see below)* | `weed_species_v3` |

`weed_species_v3` is 11-way **species classification**, not the
`rice_protect` / `weed_target` decision ontology. Its labels are not what
`agrinav data-cvat-labels` emits, and merging the two vocabularies without a
written task definition is exactly what
[`annotation_guide.md`](annotation_guide.md) warns against. Treat it as a
separate contract or leave it unmounted until someone defines it.

**Task = one split of one dataset.** Keep the split boundary in the task name:

```
rice_phase2 · train
rice_phase2 · valid
```

There is no `test` task, and there is no mount that would let you create one.

Create each task with **Connected file share**, not upload: point it at the
subdirectory under `/home/django/share`. The images stay on disk, the task
references them, and nothing is copied.

**Job = one review sitting.** Set *Segment size* when creating the task (200–300
images is a reasonable sitting). CVAT tracks stage and state per job, which is
how you see review progress without asking anyone.

The split manifest remains the authority on membership. CVAT is where images get
looked at; `manifests/split_membership.json` in the rebuilt dataset is where
membership is decided. If they disagree, the manifest wins.

## 7. Import existing annotations

To review proposals rather than draw from scratch, create the task from the
share, then **Actions → Upload annotations → COCO 1.0** and select the matching
`instances_<split>.coco.json`. CVAT matches by `file_name`, so the JSON's file
names must line up with the files in the share subdirectory the task was built
from.

Two cautions specific to this project:

- The polygon set (`.../deliverable/detection/RICE/annotations/`) is 81,204
  **unreviewed** SAM proposals. Everything imported starts at
  `review_status: unreviewed` and stays a proposal until a human changes it.
- The rebuilt phase-2 set is **boxes**, so it imports into a `rectangle`
  project — the one that needed `--allow-non-canonical-geometry`.

## 8. Export, and the conversion into records

**Actions → Export task dataset → COCO 1.0** gets the work out of CVAT.

That export is not ingestible on its own. `agrinav data-validate` — the gate that
decides what may enter a split — reads JSONL in `agrinav.annotation_record.v1`
(`data/schemas/annotation_record.v1.schema.json`), not COCO. Convert first:

```bash
agrinav data-cvat-to-records \
  --coco export/annotations/instances_default.json \
  --manifest RICE_phase2_rebuild_2026-07-29/manifests/split_membership.json \
  --dataset-id rice_phase2 --dataset-version rebuild-2026-07-29 \
  --review-metadata artifacts/cvat/review_metadata.json \
  --out artifacts/cvat/rice_phase2_reviewed.jsonl
```

then run the gate (the converter runs it too, unless `--skip-validation`):

```bash
agrinav data-validate artifacts/cvat/rice_phase2_reviewed.jsonl \
  --ontology data/ontology.v1.json --check-split-overlap \
  --manifest RICE_phase2_rebuild_2026-07-29/manifests/split_membership.json
```

Shape translation is mechanical, because the label spec above was generated from
the same ontology: every CVAT attribute maps onto a schema field, the empty
default is the `null`, and the enums are the schema's own. What the converter has
to supply is everything COCO does not carry, and each of those is a decision
documented in `src/agrinav/data/cvat_export_to_records.py`:

| Field | Source |
| --- | --- |
| `record_id` | derived: `dataset_id` + a digest of dataset/version/image/hash. Stable across re-exports; no CVAT ids, timestamps, or annotation content go into it |
| `image_id`, `source.*` | `--manifest` — the manifest wins over the COCO file. `source_image_sha256` is its `sha256` (the bytes reviewed), not `source_sha256` |
| `source.split` | manifest, with `valid` renamed to the schema's `validation` |
| `review.*` | `--review-metadata` sidecar from the CVAT REST API; absent means explicit nulls and `review_status: unreviewed` |
| `provenance.proposal_method` | `--proposal-method` (default `imported`, keeping the raw COCO objects as `original_proposal`) |
| `provenance.human_edit_state` | derived summary of the per-object `human_edit_action` values — a different field with a different vocabulary |

Two limits worth knowing before you rely on it:

- **CVAT's COCO 1.0 export drops the `image_review` tag**, because it writes
  `coco_instances` only. That tag is where `review_status`, `verified_empty` and
  `unusable` live, so without the sidecar every converted record is `unreviewed`
  and every `verified_empty` is `null`. A package nobody supplied review metadata
  for is reviewed work in a holding format, not truth — no matter how finished it
  looks in the UI. The converter will not upgrade a review state, and an export
  with no shapes never becomes a verified empty.
- **`annotation_confidence` has no CVAT default** (§5, rule 3) but the wire format
  has no null for it, so an object whose annotator never set it is a hard error
  naming the image and the object. Either the annotation guide requires the field
  in review, or the ontology needs an explicit "unstated" member; the converter
  will not invent one.

The sidecar (`agrinav.cvat_review_metadata.v1`) is written by hand or from the
REST API, and is rejected if it carries a key the converter does not know:

```json
{
  "schema_version": "agrinav.cvat_review_metadata.v1",
  "defaults": {"guide_version": "v1", "annotation_version": "cvat-task-31-v2"},
  "jobs": [{"job_id": 31, "images": ["a.jpg", "b.jpg"],
            "annotator_id": "cvat-user-4", "annotator_completed_at": "2026-08-28T09:00:00Z",
            "reviewer_id": "cvat-user-1", "reviewed_at": "2026-08-29T11:30:00Z",
            "review_status": "accepted"}],
  "images": {"b.jpg": {"verified_empty": true}}
}
```

## 9. Not done yet

- **No scripted export of the review sidecar** (§8). The job stage, assignee and
  reviewer are in the CVAT REST API, but writing `review_metadata.json` from it
  is still manual, and it is the one input that decides whether a converted
  package can be truth at all.
- **No scripted task creation.** Tasks are created in the UI for now. CVAT ships
  a Python SDK and `cvat-cli` for scripting this; adding it is worthwhile once
  the manual flow is proven. <https://docs.cvat.ai/docs/api_sdk/>
- **No model-assisted annotation.** CVAT can serve SAM and other models through
  its serverless (Nuclio) components for interactive segmentation. That is a
  separate stack with its own GPU expectations. Proposals are generated
  out-of-band instead, by `agrinav data-sam-mask` on the local RTX 4070
  (pinned to `facebook/sam2.1-hiera-large @ 665f8e2a`), and imported as COCO.
- **No licensing decision for publishing imagery.** Unchanged from the intake:
  the RICE imagery's redistribution terms have not been settled. Running CVAT
  locally does not publish anything; sharing a CVAT instance would.
