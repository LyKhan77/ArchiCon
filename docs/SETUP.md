# Setup and source checkpoint

The repository contains implementation, tests, shared configuration and operational documentation. Private machine profiles, videos, model weights/vendor cache, runtimes, job evidence, final products and local working plans are intentionally excluded. The existing configured workspace already has these files; do not overwrite its active profiles when updating source.

## Local development

Use Python3.12 and a Node/npm runtime that supports the pinned web dependencies. RTK is the shell wrapper used in project commands; it is not an application dependency. A shell without RTK can run the command following `rtk proxy` directly.

```sh
rtk proxy git clone https://github.com/LyKhan77/ArchiCon.git
rtk proxy python3.12 -m venv ArchiCon/.venv
```

Change into `ArchiCon/`, then install the local implementation/audit dependencies and the locked web dependencies:

```sh
rtk proxy .venv/bin/python -m pip install -e '.[audit]'
rtk proxy npm ci --prefix web
rtk proxy cp -n configs/paths.local.example.yaml configs/paths.local.yaml
rtk proxy cp -n configs/paths.server.example.yaml configs/paths.server.yaml
```

Edit the copied profiles. Set the absolute local/server checkout roots, a working SSH alias/user@host, and the exact authorized GPU UUID. Example placeholders are deliberately not usable resource identities. Keep profiles ignored and credentials in the system SSH/credential configuration; do not put passwords or tokens in YAML.

Python loads source under `src/`; use the documented CLI entrypoints or set PYTHONPATH for tests:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
rtk proxy node --test web/tour/calibration.test.mjs
rtk proxy .venv/bin/python scripts/serve_dashboard.py
```

Open http://127.0.0.1:8765. Source tests use temporary fixtures; backend-specific integrations and optional historical evidence tests may skip when their local prerequisites are absent. Source verification does not require a GPU reconstruction.

## Full server pipeline

Provision audit/camera, inference and geometry runtimes separately. Read [environments/README.md](../environments/README.md) and [TECHSTACK](TECHSTACK.md) before installing dependencies: the validated server uses distinct NumPy/PyTorch/Open3D environments. The inference/geometry bootstrap is not a hermetic one-command installer.

The server needs the pinned DA3 vendor source/checkpoint, masking weights, model-cache paths and authorized device from its active profile. Historical legacy aliases refer to separately provisioned read-only sources; examples do not include those datasets. Verify required paths/runtime/model provenance before running. Do not assume cloning GitHub downloads private data or model weights.

After both profiles and server runtimes are ready, follow [WORKFLOW](../WORKFLOW.md) or [ARCHICON](ARCHICON.md). New video jobs generate capture/export configuration from their own input; never reuse baseline scene caches for an unrelated recording.

## Git boundary

Publish source, tests, shared settings, example profiles, core docs and required dependency/font licenses. `.gitignore` excludes active profiles, secrets, runtime environments, node_modules, media/models, generated outputs and local planning directories. `.sync-files` is a source-transfer allowlist, not a list of files to commit: it also names the ignored active profiles after local setup.

The following stay on the configured machines:

- `raw_videos/`, cached `models/`, local `vendor/` and runtime environments.
- `outputs/jobs/`, `outputs/archive/`, QA/review evidence and `end_product/JOB/vNNN/`.
- `configs/paths.local.yaml`, `configs/paths.server.yaml` and credentials.
- Working plans/specifications in `docs/superpowers/` and archived research plans in `docs/archive/`.

References marked **lokal** in documentation identify evidence or records in an existing workspace; they are not broken promises of files in the Git repository. Keep these files immutable or versioned according to [AGENTS.md](../AGENTS.md) and WORKFLOW. Do not force-add them to make links work. Current source state and limitations are recorded in [CHANGELOG](../CHANGELOG.md).
