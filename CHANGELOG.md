# Changelog

Completed changes are recorded here. Versions describe source checkpoints; private reconstruction products keep their independent `vNNN` delivery versions.

## [Unreleased]

### Global architecture alignment (in progress)

- Added conservative camera-world similarity fitting and a deterministic alignment graph. Six shared cameras are required, with disjoint fit/validation cameras, position/orientation gates and cycle checks. Unsupported worlds remain separate; alignment does not validate metric scale or fill missing surfaces.

### Changed

- The default room-tour profile now samples original video frames adaptively: base 2 FPS and up to 4 FPS in high-motion intervals, with sequential camera matching at overlap 32. Loop closure is optional and requires a local vocabulary tree for offline execution.
- Audit reports persist motion metrics and sampling settings so higher-density frame selection remains traceable and bounded.
- The default motion threshold is calibrated to the recorded iPhone profile at 0.03; it remains configurable per job because motion scores depend on resolution and scene content.

### Fixed

- DA3 smoke/full-depth resource isolation now accepts the configured GPU by UUID regardless of its host index. The selected device remains the only visible CUDA device and is addressed as logical `cuda:0`.
- Added regression coverage for a selected physical GPU at index 1.

## [0.1.0] — 2026-10-19

First source checkpoint of ArchiCon. The working reconstruction/dashboard/viewer implementation was validated during 14–16 October 2026; this checkpoint adds repository guidance and publication hygiene.

### Added

- Video-to-room pipeline: orientation/PTS audit, keyframe selection, CPU SfM cameras, pose-conditioned DA3-LARGE depth, masks/filtering, CPU TSDF, mesh/render and draft floorplan export.
- End-to-end orchestration with detached worker/controller, resource gates, checksums, resume, immutable attempts and versioned delivery.
- ArchiCon local dark dashboard for input/upload, actual progress/logs/previews, result viewing and ZIP download.
- Offline architecture/interior room tour with embedded Plex/fonts/assets, component thumbnails, orbit, roof clipping, recorded-camera playback/gap stops, 2D evidence and draft calibration JSON/SVG.
- Manual review packet/schema and bounded S06/S07 refinement; no automatic review-service invocation.
- Reversible cold archival with SHA inventories, no-overwrite renames, lock checks, journal and restore.
- AGENTS.md repository rules, safe machine-profile examples and setup documentation. Private profiles, runtime/data/model files and working plans are excluded from Git.

### Verified baseline

- Fresh full-video, smoke, nested-layout and HTTP packaging paths have recorded local/server evidence.
- The independent room recording completed; the user reported overall OK. Its private v002 delivery aligns the viewer with ArchiCon; mesh/floorplan/camera data are identical to v001, with quality review still pending/unassessed.
- Prior viewer regression: 258 Python tests (254 passed, 4 environment-dependent skips), 7 JavaScript calibration tests and 65 server pipeline tests. Offline QA covered all 4 components, 252 unique recorded views, 320–1440 px layouts, playback/gap/lookaround, JSON/SVG, HTTP/CSP, dashboard iframe and ZIP.
- Checkpoint regression: 258 Python tests in the workspace (254 passed, 4 backend-dependent skips) and in a clean staged-source export (253 passed, 5 skips, including one unavailable private audit fixture). Both runs passed all 7 JavaScript calibration tests. The clean export reused the existing Python environment; it was not a fresh dependency installation.
- Frozen offline `npm ci --prefix web` succeeded from the clean export root without changing the lockfile. Publication checks cover 150 source/documentation files and exclude private profiles, secrets, media, reconstruction products, model weights and local plans.

### Known limitations

- Dimensions are relative until a measured reference is calibrated and independently checked. Floorplans are geometry slices, not semantic architectural drawings or survey measurements.
- Frame coverage does not prove complete room/surface coverage. Holes, glass/dynamic-object errors and separate unaligned component worlds remain possible.
- Review is manual; completed computation is not quality acceptance. Unobserved geometry requires better capture rather than invented surfaces.
- MapAnything, optimized Gaussian Splatting, generic camera recovery/submap alignment and CAD export are not implemented.
- The full pipeline requires a configured server, local DA3/vendor/model cache and separate audit/inference/geometry runtimes. Runtime setup is documented but not a hermetic one-command deployment.
- Final HTML, videos, meshes and historical job evidence remain on the configured devices/server; cloning this repository does not retrieve them. Cold experiments must be restored before historical resume/reuse.
