# Entrypoint pengguna

Gunakan `run_pipeline.py run --input VIDEO --job-id ID` dari root proyek. Job baru: `outputs/jobs/ID/`; hasil final: `end_product/ID/vNNN/`. Subcommand: plan, status, resume, prepare-review, apply-review. Review manual saja. Lihat [README](../README.md) dan [WORKFLOW](../WORKFLOW.md).

Stage CLI berikut memakai run-ID resolver yang mendukung direktori legacy maupun stage bersarang. Untuk penggunaan umum, jalankan orchestrator agar resource, transfer, checksum dan QA dikelola bersama.

Eksperimen closed kini berada di arsip dingin. `archive_outputs.py verify|restore|apply` memeriksa SHA dan memindahkan kelompok sesuai manifest, tanpa fungsi hapus. Resume/reuse eksperimen lama memerlukan restore; lihat [panduan arsip](../docs/OUTPUTS_ARCHIVE.md). Pipeline fresh tidak membutuhkan arsip.

# Scripts

Jalankan dari root checkout dengan `.venv/bin/python`:

| Entrypoint | Hasil |
|---|---|
| `preflight.py --config configs/paths.server.yaml --run-id NAME` | Inventory, source provenance, resource gate, validasi file cache; tanpa inference |
| `audit_video.py --run-id NAME --threads 4` | Decoder PTS, rotasi, sharpness/exposure, uniform/adaptive keyframe asli, split temporal, contact sheets |
| `import_legacy.py --run-id NAME` | Manifest dan review cached DA3; timestamp perkiraan, visual-only |
| `localize_validation.py --audit-run AUDIT --camera-run CAMERA --run-id NAME` | Query-only PnP delapan validation view; fixed map/K, no BA; OPENCV remap pada grid train |
| `smoke_da3.py --camera-run CAMERA --run-id NAME` | Empat train view DA3-LARGE lokal/offline; relative Z-depth, camera contract, peak memory |
| `reconstruct_cameras.py --audit-run AUDIT --run-id NAME` | Kamera pilot CPU train-only, residual native pixels, RGB undistorted dan K baru |

Setiap run harus memakai ID baru. `status.json` menentukan tahap selesai/gagal;
keberadaan artefak saja tidak cukup. Kamera hanya menerima audit dengan status
`video_audit/complete`. Default rentang kamera 15–45 detik, OPENCV, empat thread,
feature extraction longest edge 960. Input kamera disalin pada resolusi asli.
Gate saat ini: registration ≥90%, median residual ≤1,5 px dan P95 ≤4 px pada
resolusi asli. Validation dikecualikan dari map training; entrypoint localization hanya mengestimasi
query pose pada map tetap.

Smoke depth hanya menerima `.venv-inference` server dengan NumPy1.26.4 dan vendored
DA3. Gunakan supervisor artefak run untuk abort RSS/GPU-process budget saat runtime;
CLI sendiri membatasi allocator PyTorch dan memeriksa host RSS setelah selesai.
Splatting belum tersedia; viewer mesh HTML offline tersedia melalui exporter di bawah.

Pilot surface entrypoints kini tersedia:

| Entrypoint | Kontrak |
|---|---|
| `run_depth_pilot.py --run-id NAME [--max-windows 1]` | Empat window 16/stride 12 dari52 TRAIN, raw64 + selected52; satu window hanya resource proof |
| `make_masks.py --run-id NAME` | CPU SSDLite person boxes + manual glazing/person annotations, grid280×504; incomplete |
| `filter_depth.py --depth-run DEPTH --mask-run MASKS --run-id NAME` | Spatial/MV masks, occlusion/contradiction/support, camera-facing normals, coverage |
| `fuse_tsdf.py --filter-run FILTER --run-id NAME` | Dua colored surface PLY/GLB, shared relative parameters; requires geometry runtime/Open3D0.19 |
| `filter_depth.py ... --confidence-quantile 0` | q0 comparison; defaultq10; input/camera/masks identik |
| `run_fusion_ablation.py --no-confidence-filter-run Q0 --variant VARIANT --run-id NAME` | Satu variasi CPU; reference parameters tetap, paired provenance wajib; tanpa inference |
| `audit_mesh_holes.py --mesh-run MESH --views-run FIXED_AUDIT --run-id NAME` | Triangle raycasting one/two-sided, fixed8query+exterior, boundary diagnostics; bukan accuracy score |

Runtime budgets dipantau supervisor artefak run; original model, input, dan source
lama tidak diubah. Filter/fusion memerlukan tahap upstream selesai serta52 TRAIN
artifacts yang cocok hash/camera. Kedua TSDF menyimpan semua components; tidak
melakukan smoothing, hole fill, pruning, atau klaim collision readiness.
Pilihan variant: `voxel1p5_trunc4`, `voxel2_trunc4`, `voxel1p5_trunc8`,
`voxel2_trunc8`, `confidence0_voxel2_trunc4`. Satu process per variant agar RSS
dilepas. Audit khusus global kamera/map pilot002; menolak world lain. Pakai CPU
watchdog artefak seperti dijelaskan di README root.

Full-video entrypoints:

| Entrypoint | Kontrak |
|---|---|
| `reconstruct_full_cameras.py --audit-run AUDIT --run-id NAME` | Semua audited frames, source split sebagai metadata produksi; preserve separate model worlds dan missing IDs |
| `... --matching sequential --overlap 32 --loop-detection --vocab-tree-path PATH --max-image-size 1600 --max-features 8192` | Matching temporal bounded dengan loop closure memakai vocabulary tree lokal; tidak mengunduh asset atau menyambungkan area tanpa overlap visual |
| `run_full_depth.py --camera-component outputs/RUN/components/ID --run-id NAME` | Variable-N component, max16view/stride12, endpoint coverage, actual alignment capture; --max-windows partial bukan complete |
| `build_full_surface.py --depth-run DEPTH --camera-component PATH --annotations FILE --run-id NAME` | Current-grid semantic boxes + spatial/MV filtering, coverage/strict comparison, shared unit-relative parameters; exclusive-create checkpoint |
| `render_full_surface.py --surface-run SURFACE --run-id NAME` | Seluruh accepted component poses dirender sebagai triangle, per-frame PNG,2fps MP4,overview dan display-only GLB |

`full_take.prepare_component` menolak extreme trajectory centers sebelum inference.
Pose gagal dipulihkan lewat query-only fixed-mapPnP bila strict gate lolos; tidak
mengubah binary map lama. Corrected component bersifat immutable. Paket final
all-frame timeline dibangun oleh artefak `outputs/l2bp3_full_delivery_001/build_delivery.py`,
termasuk frame gagal dan rekomendasi capture. Script watchdog aktual disimpan
di direktori delivery, bukan library dependency baru.

Ekspor denah dan tour dari rekonstruksi terverifikasi:

| Entrypoint | Kontrak |
|---|---|
| `export_room_tour.py --config configs/room_tour.json --run-id NAME` | Runtime geometry CPU; verifikasi SHA mesh/kamera terhadap render manifest, bake transform sekali, tiga irisan, garis berulang, SVG/PNG, mesh ringan, tour-data.json |
| `node web/build.mjs --data outputs/NAME/tour-data.json --output outputs/NAME/room-tour.html` | Device dengan dependency web terpasang; satu HTML offline, GLB/PNG/JS/CSS embedded, exclusive-create |

Kedua tahap ini tidak menjalankan rekonstruksi video ulang. Meter memerlukan acuan;
komponen tidak disejajarkan otomatis. Hasil gagal tetap memiliki status dan ID
lama tidak boleh ditimpa. Lintasan kamera di luar batas mesh tetap terlihat pada
denah, sedangkan label bentang geometri memakai batas mesh asli.
