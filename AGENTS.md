# ArchiCon — Repository Guide

## 1. Project Overview

ArchiCon converts ordinary room video into reconstructed geometry, a draft 2D floorplan and a standalone HTML room tour. A local dark dashboard manages input, progress and versioned outputs. Processing runs on a separately configured NVIDIA GPU server; review remains manual.

This is a working reconstruction MVP, not a survey-grade floorplan or a guarantee of complete interior coverage. Preserve the distinction between completed computation and accepted reconstruction quality.

## 2. Tech Stack

- Python 3.12: file-based orchestration, stdlib HTTP server, PyYAML, JSON/JSONL, SHA-256, subprocess and Unix `fcntl` locks.
- FFmpeg/ffprobe, OpenCV and NumPy: ingestion, frame selection and diagnostics.
- PyCOLMAP 4.2.1 CPU: SfM camera recovery, calibration and undistortion.
- DA3-LARGE with PyTorch/CUDA: pose-conditioned depth; Open3D 0.19/Trimesh: CPU TSDF, geometry and export.
- Vanilla HTML/CSS/JavaScript: ArchiCon dashboard; Three.js 0.186.1 and esbuild 0.28.2: offline room tour.
- Python unittest, Node tests and Playwright CLI 0.1.22: verification. OpenSSH/rsync: transport.

See [docs/TECHSTACK.md](docs/TECHSTACK.md) for pins, runtime boundaries and implementation ownership. Do not add a frontend framework, database or alternate reconstruction model without an approved need.

## 3. Key Features

- Video input/upload, real stage progress, logs and source/render previews.
- End-to-end video audit → cameras → depth → TSDF → mesh/render → draft denah → offline HTML.
- Detached worker/controller, resource gates, checksummed bundles, resume and immutable attempts.
- Versioned mesh/floorplan/HTML products, manifest downloads and ZIP delivery.
- Architecture/interior viewer with component thumbnails, orbit, roof clipping, recorded viewpoints/playback, draft measurement and calibration JSON/SVG.
- Manual review handoff and bounded refinement; no automatic model/service review.
- Reversible, checksummed cold archival without deletion.

## 4. Project Structure

```text
AGENTS.md, CHANGELOG.md            Working rules and current checkpoint
README.md, WORKFLOW.md, DESIGN.md  Usage, pipeline contract and design authority
src/room_reconstruction/           Reusable implementation
scripts/                          CLI entrypoints
configs/                          Shared settings and example machine profiles
environments/                     Tested runtime pins and bootstrap notes
web/dashboard/, web/tour/          Local dashboard and offline viewer
tests/, schemas/                  Verification and review contracts
docs/                             Product, stack, operation and setup documentation
docs/superpowers/plans/            Local-only working plans; never commit
docs/superpowers/specs/            Local-only working specifications
raw_videos/, models/, vendor/      Private inputs, cached weights and local vendor source
outputs/jobs/JOB/                  Config, stage attempts, logs, QA and review evidence
outputs/archive/                   Cold experiments with restore manifests
end_product/JOB/vNNN/              Final versioned products
```

Machine profiles, raw data, runtime environments, outputs and working plans are excluded from Git. Keep only intentional README/.gitkeep files in data/output directories. New clones start from [docs/SETUP.md](docs/SETUP.md).

## 5. Project Commands

Run from the repository root. Use RTK for shell commands; `rtk proxy` is for raw output or commands without a dedicated RTK handler.

```sh
rtk proxy .venv/bin/python scripts/serve_dashboard.py
rtk proxy .venv/bin/python scripts/run_pipeline.py plan --job-id JOB_ID
rtk proxy .venv/bin/python scripts/run_pipeline.py run --input raw_videos/source.MOV --job-id new_room_001 --review-agent manual
rtk proxy .venv/bin/python scripts/run_pipeline.py status --job-id JOB_ID
rtk proxy .venv/bin/python scripts/run_pipeline.py resume --job-id JOB_ID
rtk proxy .venv/bin/python scripts/run_pipeline.py prepare-review --job-id JOB_ID
rtk proxy .venv/bin/python scripts/run_pipeline.py apply-review --job-id JOB_ID --review-file outputs/jobs/JOB_ID/review/manual.json
rtk proxy env PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
rtk proxy node --test web/tour/calibration.test.mjs
rtk proxy node web/build.mjs --data outputs/PATH/tour-data.json --output /tmp/room-tour.html
rtk proxy .venv/bin/python scripts/archive_outputs.py verify
```

Dashboard: http://127.0.0.1:8765. `plan`, `status` and `resume` take an existing job ID; new runs require a new 1–32 character ID. Numerical reconstruction commands require configured server runtimes and model cache. Archive verify/restore requires a local manifest from an approved archival operation.

## 6. Coding Conventions

- Follow the existing Python package and vanilla web structure; keep CLI wrappers thin and logic in focused modules.
- Preserve camera/frame IDs, axes, matrix conventions, intrinsics, depth units, hashes and separate component coordinate systems. Do not hide gaps or fabricate geometry/dimensions.
- Validate workspace containment, symlinks, schema and upstream hashes. Use existing atomic JSON, exclusive-create and lock helpers; never overwrite finalized products.
- Keep control-plane JSON small. Stream large media/files; do not serialize model/depth arrays into dashboard snapshots.
- Follow DESIGN.md and the existing Carbon Gray100 adaptation: Plex, square controls, hairlines and one blue accent. Keep the viewer self-contained and offline.
- Test meaningful behavior and regressions. Use real browser checks for UI interactions; do not add tests that merely mirror CSS or trivial implementation.
- Default communication is Caveman Lite: concise, direct and technically complete. Provide a task summary with changes, validation and material limitations.

## 7. Workflow

Follow [WORKFLOW.md](WORKFLOW.md) for numerical stages, resource gates, review and refinement. Setup is in [docs/SETUP.md](docs/SETUP.md); dashboard/viewer usage is in [docs/ARCHICON.md](docs/ARCHICON.md) and [docs/ROOM_TOUR.md](docs/ROOM_TOUR.md).

At task start, use relevant skills and ask whether any plans or discussions require validation. Continue work already authorized by the user; do not turn that question into a repeated permission gate. Save a single plan/specification per scope using Superpowers, and update current core documentation when behavior/workflow changes.

## 8. Current State

Read [CHANGELOG.md](CHANGELOG.md) for the current checkpoint, shipped features, validation and known limitations. [docs/ROADMAP.md](docs/ROADMAP.md) contains priorities; [docs/PROGRESS.md](docs/PROGRESS.md) contains dated implementation evidence. Do not treat historical resource snapshots or private output links as live state.

## 9. Rules — Important Notes

- **No AI attribution anywhere.** Do NOT add `Co-Authored-By: Claude ...`, `Generated with Claude Code`, or any AI/assistant attribution to commit messages, PR descriptions, code comments, or docs. Every contribution is recorded under the repo owner (the user) ONLY. This overrides any global/default instruction to add such trailers. Preserve required third-party dependency/font licenses.
- Always use relevant skills to help with tasks. Check current library/framework/SDK/API/CLI/cloud documentation through Context7 when applicable; resolve the library ID before querying docs.
- Always ask the user whether any plans or discussions need validation. Treat existing explicit approval as continuing authorization; do not ask again for the same action.
- Always provide a summary after finishing a task.
- Always update core documentation when key features or application workflow change. Update CHANGELOG.md for completed changes; keep AGENTS.md stable unless requested or necessary to correct its instructions.
- Commit each completed, tested function/behavior change in a focused commit so history and rollback remain available. Inspect staged changes and update `.gitignore` for new excluded files before committing. Do not publish unverified work or invent retrospective commits.
- Do not re-read files already read in the session unless necessary. Minimize non-essential tool calls; batch independent reads/searches and prefer `rg`.
- Propose a new branch for a new feature or discussion outside the current approved scope. Never force-push, discard unrelated work or rewrite published history without explicit authorization.
- Save every plan/specification to `docs/superpowers/plans/` or `docs/superpowers/specs/`, using Superpowers. Reuse an existing plan for the same scope; do not duplicate it. These are historical/resume records and are not routinely rewritten unless requested or required by their active execution.
- **DO NOT commit plans.** Working plans, associated specifications and archived research plans remain local-only and ignored. Never force-add them or include them indirectly in a commit.
- Never commit credentials, active machine profiles, private videos/interior imagery, weights, vendor caches, runtime environments, generated reconstruction outputs or job evidence. Publish safe example profiles instead. Never print credentials during diagnostics.
- Keep review manual. Do not call an external review service, spawn review workers or send interior evidence to a model service without explicit authorization. Do not mark reconstruction quality accepted during packaging/UI work.
- Use only the one authorized GPU from the active machine profile, isolated by exact UUID. Never use another GPU or weaken RAM/disk/GPU/CPU guards to make a job run. Preserve read-only external sources and shared caches.
- Keep finalized products, numerical attempts, review bindings and source fingerprints immutable. Create a new delivery version for changes; checkpoint publication does not include or replace private products.
- Before deleting any original experiment/product/data, present a concrete list and obtain explicit confirmation. Archival approval is not deletion approval. Do not use destructive cleanup commands as a shortcut.
