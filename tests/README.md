# Verification scope

Checkpoint source0.1.0 (19Oktober2026):258tes lokal,254lulus/4skip backend;7tes JS kalibrasi. Jalankan suite dari checkout bersih setelah setup Node/Python; tes backend atau bukti audit privat dapat skip bila prasyaratnya tidak tersedia. Snapshot angka166di bawah menyimpan riwayat fase sebelumnya. Panduan checkout: [SETUP](../docs/SETUP.md); perubahan terbaru: [CHANGELOG](../CHANGELOG.md).

```bash
rtk proxy env PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Suite menguji resource gate/GPU remapping, provenance dan run collision, decoder
PTS/rotasi termasuk fixture VFR, frame selection/split, import cache legacy,
pose inversion/reprojection, intrinsics resize, residual dari model distorsi,
status failure, penolakan upstream yang belum selesai, fixed-map query PnP,
OpenCV thread caps, serta kontrak train-only DA3/processed K/Z-depth.

Tes unit memakai fixture dan tidak memerlukan GPU atau PyCOLMAP lokal. Integrasi
PyCOLMAP nyata dijalankan terpisah pada server; laporan tersimpan per run.
166 tes terverifikasi: server geometry seluruhnya lulus; laptop163 lulus dan
tiga tes backend PyCOLMAP/Open3D dilewati. Load/inference DA3 nyata diverifikasi
terpisah pada smoke empat view.
Tes baru mencakup overlapping windows, semantic-mask rasterization, occlusion vs
contradiction, projection bounds, camera-facing normals, unit Z dan world-plane
TSDF nyata, rejected/stale backend serta fixed-camera fusion. Ekspor mesh aktual
diverifikasi terpisah; suite tidak membuktikan akurasi metric atau kernel splatting.
Tambahan fase holes: nearest front-facing triangle di belakang backface, optical-Z
ray parameters, GLB binary/material preservation, fixed-map/calibration hash
rejection, controlled q10/q0 provenance dan fixed reference TSDF parameters.
Full-video contracts menambah all-frame selection, separate-component accounting,
variable16-view windows/endpoints, production split metadata, trajectory outlier,
strict/uncertain coverage, numericcamera schema dan actual exclusive-create JSON
control-flow regression. Video/GLB aktual diverifikasi melalui artefak server.
