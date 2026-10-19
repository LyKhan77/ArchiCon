# Catatan pengembangan dan eksperimen historis

**Ditinjau:** 16 Oktober 2026. Dokumen ini menyimpan eksperimen dan command stage sebelum layout job baru; angka hardware/test/resource adalah snapshot historis. Untuk video baru, ikuti [ARCHICON](ARCHICON.md), [README utama](../README.md) dan [WORKFLOW](../WORKFLOW.md). Stack aktual berada di [TECHSTACK](TECHSTACK.md); pengguna melaporkan uji baru selesai, overall OK; review formal tetap pending. Path asal pada contoh command historis memerlukan [restore arsip](OUTPUTS_ARCHIVE.md); tautan laporan mengarah ke arsip. Sebagian artefak numerik hanya tersimpan di server.

# Room Reconstruction — Part 2

Pipeline dari video iPhone ke kamera, DA3 depth, mesh, denah draft dan tour HTML.
Seluruh timeline video acuan sudah dicoba. Hasil fresh terbaru:283/302keyframe,
19gap. Arsip baseline sebelumnya:286/302 dengan entrance-local terpisah.
Optimized surface/Gaussian Splatting belum dijalankan. Entrypoint
`scripts/run_pipeline.py` mengatur video→output+QA, lalu menunggu review manual.
Regresi cache, smoke fresh60/60 dan cold-run video penuh sudah berhasil sampai HTML+QA.
Review manual tetap diperlukan untuk kualitas geometri; rekaman independen belum diuji.
Kebutuhan: [PRD.md](PRD.md), tahapan:
[ROADMAP.md](ROADMAP.md), hasil aktual: [PROGRESS.md](PROGRESS.md).

Alur, command, review manual dan refinement: [WORKFLOW.md](../WORKFLOW.md).
Pipeline tidak memanggil agent/LLM otomatis, sesuai kebijakan usage pengguna.

```bash
rtk proxy .venv/bin/python scripts/run_pipeline.py run --input raw_videos/rekaman_baru.MOV --job-id rekaman_baru_001
rtk proxy .venv/bin/python scripts/run_pipeline.py status --job-id rekaman_baru_001
rtk proxy .venv/bin/python scripts/run_pipeline.py resume --job-id rekaman_baru_001
```

Output job baru sekarang: `end_product/<job>/vNNN/`; path `end_product/part2/` di contoh hasil adalah arsip. Review pending sampai pengguna
menjalankan review dan menerapkan JSON lewat `apply-review`; lihat WORKFLOW.

## Workspace

| Lokasi | Peran |
|---|---|
| `/absolute/path/to/ArchiCon` | Source utama, konfigurasi, tes ringan, review |
| `/srv/archicon` | Eksekusi server dan artefak pipeline |
| `/srv/archicon-legacy` | Baseline dan video lama, read-only |
| `/srv/model-cache/huggingface/hub` | Snapshot checkpoint lama, read-only |

Kedua checkout memakai branch `setup/project-structure`, belum ada commit/remote.
Sinkronisasi source memakai allowlist `.sync-files`; `.git` tetap lokal tiap mesin.
Video server adalah symlink ke file lama dengan SHA-256 identik. Kebijakan read-only
pada symlink bersifat aplikasi, bukan mount read-only OS.

## Denah 2D dan tour HTML

Paket hasil PART1: `end_product/part1/` (`../end_product/part1/README.md`, lokal), berisi
mesh detail, denah SVG/PNG dan `room_tour/index.html` yang dapat dibuka offline.

Hasil fresh terbaru: HTML offline (`../end_product/part2/part2_full_002/v002/room_tour/index.html`, lokal),
manifest (`../end_product/part2/part2_full_002/v002/manifest.json`, lokal),
review dan saran capture (`../outputs/archive/experiments_2026-10-16/flat/part2_full_002/MANUAL_REVIEW.md`, lokal).
Review manual selesai: `needs_capture`; 283/302keyframe dalam satu world utama,
19gap, skala relatif. Kamera/depth/mesh baru; hanya audit fresh full001 dipakai ulang.
Cold-run seluruh video tanpa reuse rekonstruksi: `part2_full_001`,18stage sampai
HTML+QA. Perbandingan matcher (`../outputs/archive/experiments_2026-10-16/flat/part2_full_002/camera-comparison.json`, lokal).

Arsip baseline dengan entrance terpisah tetap tersedia:
HTML286/302 (`../end_product/part2/part2_cache_001/v003/room_tour/index.html`, lokal).
Hasil fresh belum memulihkan tiga view entrance-local tersebut; tidak mengklaim
coverage lebih lengkap daripada arsip itu. PART1 dan semua versi lama tetap utuh.

Hasil exporter PART1: room-tour.html (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_room_tour_005/room-tour.html`, lokal), satu file
HTML offline; buka langsung di browser. Berisi orbit, potongan atap,283kamera utama
dan3kamera pintu masuk, look-around, playback yang berhenti pada gap, denah,
pengukuran dua titik, kalibrasi per-komponen dan unduh SVG.

Denah SVG utama (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_room_tour_005/main-floorplan.svg`, lokal) ·
PNG (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_room_tour_005/main-floorplan.png`, lokal) ·
Denah pintu masuk terpisah (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_room_tour_005/entrance-floorplan.svg`, lokal).
Garis berasal dari tiga irisan horizontal pada mesh aktual. Furnitur/partisi masih
dapat muncul; ruangan tertutup dan luas ruangan tidak diasumsikan. Bentang geometri
X/Z termasuk outlier, bukan ukuran kamar. Arah atas/lantai masih estimasi.

Belum ada ukuran nyata. Pada tab **Denah 2D**, pilih dua titik dengan jarak yang
nantinya diukur di lokasi, isi panjang dalam meter, lalu **Terapkan skala**.
Label `u` berubah menjadi `≈m`. Pada HTML PART2, simpan/muat kalibrasi melalui JSON
yang terikat SHA mesh; SVG unduhan menyimpan skala draft dan segmen pilihan.
Komponen pintu masuk perlu kalibrasi sendiri. HTML PART1 lama tetap immutable.
Satu acuan memberi skala; pengukuran lain diperlukan untuk memeriksa distorsi.

Ekspor ulang dari hasil rekonstruksi yang diterima, dengan ID baru:

```bash
# Server, CPU; tidak menjalankan DA3 ulang atau memakai GPU.
rtk proxy env CUDA_VISIBLE_DEVICES= .venv-geometry/bin/python scripts/export_room_tour.py --config configs/room_tour.json --run-id room_tour_NEW
# Device: setelah hasil ekspor disalin ke outputs/room_tour_NEW.
rtk proxy npm ci --prefix web
rtk proxy node web/build.mjs --data outputs/room_tour_NEW/tour-data.json --output outputs/room_tour_NEW/room-tour.html
```

Command di atas khusus exporter. Untuk video→output gunakan `scripts/run_pipeline.py`.
Mesh viewer utama585.455segitiga (~11,5MB), turunan ringan dari3.964.088segitiga.
GLB detail asli tetap di `outputs/l2bp3_full_delivery_001/room-full.glb`.

## Tech stack yang dipakai

Rujukan tunggal versi/runtime dan pemetaan kode adalah [TECHSTACK.md](TECHSTACK.md). Kamera full pipeline memakai PyCOLMAP CPU; fixed-map PnP historis merupakan pemulihan khusus data acuan, bukan recovery otomatis generik untuk setiap video.

MapAnything dan Gaussian Splatting belum menjadi bagian hasil yang tersedia.
Room tour hasil ekspor berjalan tanpa backend, CDN, atau koneksi internet. Dashboard ArchiCon memerlukan backend localhost yang dijelaskan pada ARCHICON.md.

```text
3d-reconstruction_part2/
├── PRD.md / ROADMAP.md / PROGRESS.md / README.md
├── configs/                       # Mesin, capture, pinned models, rencana pilot
├── environments/                  # Pin dependency CPU yang telah diuji
├── scripts/
│   ├── preflight.py
│   ├── audit_video.py
│   ├── import_legacy.py
│   ├── reconstruct_cameras.py
│   ├── localize_validation.py
│   ├── smoke_da3.py
│   ├── run_depth_pilot.py
│   ├── make_masks.py
│   ├── filter_depth.py
│   └── fuse_tsdf.py
├── src/room_reconstruction/        # Provenance, resources, ingestion, legacy, poses
├── tests/
├── raw_videos/lantai2_B_part3.MOV
├── models/huggingface -> shared cache       # Server
├── data/legacy/{l2bp3_stream,l2bp3_viewer}   # Symlink server
└── outputs/<run_id>/               # Artefak dan status; tidak masuk Git
```

## Perintah lokal

Gunakan `.venv/bin/python` Python 3.12; Python default laptop 3.14. Dependency audit
sudah terpasang. Package belum dipasang editable; entrypoint mengatur src path.

```bash
rtk proxy env PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
rtk proxy .venv/bin/python scripts/audit_video.py --run-id l2bp3_audit_NEW --threads 4
```

## Perintah server

Jalankan di root checkout server melalui SSH. Nama run berikut adalah contoh baru;
setiap ID hanya boleh digunakan sekali.

```bash
rtk proxy ssh gpu-server 'cd /srv/archicon && .venv/bin/python scripts/preflight.py --config configs/paths.server.yaml --run-id preflight_NEW'
rtk proxy ssh gpu-server 'cd /srv/archicon && CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/import_legacy.py --run-id legacy_NEW'
rtk proxy ssh gpu-server 'cd /srv/archicon && CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/reconstruct_cameras.py --audit-run l2bp3_audit_local_002 --run-id camera_NEW'
```

RTK berjalan di laptop. Camera stage CPU memakai maksimal empat thread dan
mengecek RAM/disk; PyCOLMAP 4.2.1 tersedia di `.venv` server. `status.json` menentukan
hasil stage. Resource pass belum menyatakan GPU inference siap. Sebelum pekerjaan
GPU, ulangi preflight dan isolasi GPU UUID terpilih menjadi logical `cuda:0`.

## Validation dan smoke depth

Jalankan di root server. Localization memakai CPU dan fixed training map:

```bash
rtk proxy ssh gpu-server 'cd /srv/archicon && .venv/bin/python scripts/localize_validation.py --audit-run l2bp3_audit_local_002 --camera-run l2bp3_camera_pilot_002 --run-id validation_NEW'
```

Smoke DA3 memakai `.venv-inference`, source server `vendor/`, dan cache lokal.
Run aktual dijalankan dengan supervisor yang memantau RSS ≤6 GiB, GPU process
memory ≤14 GiB, serta timeout 600 detik. Script supervisor tersimpan sebagai
artefak `outputs/l2bp3_da3_smoke_001/reports/supervisor.py`; argumen pertamanya ID
run baru. `smoke_da3.py` sendiri mengecek resource sebelum import CUDA dan memberi
PyTorch memory cap 12 GiB; host peak RSS diperiksa setelah inference.

Runtime inference mewarisi package sistem/user yang sama seperti runtime lama;
lihat [environments/README.md](../environments/README.md) untuk batas reproduksinya.

## Pilot depth, filter, dan TSDF

Hasil terbaru mencakup seluruh timeline video, dengan gap kamera ditandai:

- GLB main environment (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/room-full.glb`, lokal)
- GLB area masuk, world lokal terpisah (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/entrance-local.glb`, lokal)
- Video source/render seluruh take (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/full-video-review.mp4`, lokal)
- Overview (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/overview.png`, lokal) · Cakupan dan rekaman tambahan (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/report.json`, lokal)

302keyframe dicoba;286memiliki depth/geometri/render dan16gap ditampilkan eksplisit.
Ini pemrosesan seluruh video, belum semua interior lengkap. Area masuk0–7s,
interior ruang berkaca120–143s dan ujung tangga148,79–150,84s perlu capture tambahan.
GLB utama memakai kamera yang telah dikoreksi; hasil `surface_full_001` dan
`render_full_001` tidak dipilih karena satu pose ekstrem, disimpan untuk diagnosis.

GLB pilot lebih bersih, estimated Y-up (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_fusion_review_001/room-pilot-improved.glb`, lokal) ·
Alternatif coverage q0 (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_fusion_review_001/room-pilot-coverage-q0.glb`, lokal) ·
Perbandingan triangle asli (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_fusion_review_001/comparison.png`, lokal) ·
Review dan batas hasil (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_fusion_review_001/review.md`, lokal).
Kandidat voxel1,5×/trunc8/q10 mengurangi speckle dinding; mesh tetap open dan warna
blur. Coverage/jumlah triangle bukan ukuran akurasi. GLB sebelumnya tetap tersedia
pada `outputs/l2bp3_tsdf_001/room-pilot.glb`.

Server memiliki `.venv-inference` untuk DA3 dan `.venv-geometry` untuk Open3D 0.19.
CLI stage historis berikut memakai run ID baru; untuk input video baru gunakan orchestrator/dashboard, bukan rangkaian pilot ini:

```bash
rtk proxy ssh gpu-server 'cd /srv/archicon && .venv-inference/bin/python scripts/run_depth_pilot.py --run-id depth_NEW'
rtk proxy ssh gpu-server 'cd /srv/archicon && .venv-inference/bin/python scripts/make_masks.py --run-id masks_NEW'
rtk proxy ssh gpu-server 'cd /srv/archicon && .venv/bin/python scripts/filter_depth.py --depth-run depth_NEW --mask-run masks_NEW --run-id filter_NEW'
rtk proxy ssh gpu-server 'cd /srv/archicon && .venv-geometry/bin/python scripts/fuse_tsdf.py --filter-run filter_NEW --run-id mesh_NEW'
```

Run aktual DA3/TSDF diluncurkan melalui supervisor artefak dengan batas RAM 6 GiB
saat runtime, GPU process 14 GiB untuk DA3, dan timeout 600 detik. CLI menyimpan
peak/resource sendiri; gunakan supervisor untuk abort RSS saat proses masih hidup.
Mask manual pada `configs/masks_l2bp3.json` dipin ke image hashes; bukan segmentasi
sempurna untuk video lain. TSDF memakai `depth_scale=1` dan unit scene relatif.

Controlled fusion/audit tanpa inference ulang, jalankan dari root server:

```bash
rtk proxy .venv-geometry/bin/python scripts/filter_depth.py --depth-run l2bp3_depth_pilot_001 --mask-run l2bp3_masks_002 --confidence-quantile 0 --run-id q0_NEW
rtk proxy .venv-geometry/bin/python scripts/run_fusion_ablation.py --no-confidence-filter-run q0_NEW --variant voxel1p5_trunc8 --run-id variant_NEW
rtk proxy .venv-geometry/bin/python scripts/audit_mesh_holes.py --mesh-run variant_NEW --views-run l2bp3_holes_reference_002 --run-id visibility_NEW
```

Watchdog CPU eksperimen tersedia pada
`outputs/l2bp3_fusion_review_001/cpu_supervisor.py`; argumen: run ID baru,
path script yang diizinkan, lalu CLI args tanpa `--run-id`. Watchdog membatasi
RAM6GiB, CPU4thread, CUDA hidden, timeout600detik; CLI mengecek resource sebelum
import backend. Raycast menyimpan per-view depth/RGB/hit dan boundary diagnostics.
Delapan query kini untuk development selection; evaluasi akhir perlu unseen views.

## Sinkronisasi source

```bash
rtk proxy rsync -av --dry-run --files-from=.sync-files ./ gpu-server:/srv/archicon/
rtk proxy rsync -av --files-from=.sync-files ./ gpu-server:/srv/archicon/
```

Tambahkan source baru ke `.sync-files`. Allowlist mengecualikan `server-info.txt`,
`.git`, environment, video, model, baseline, dan output. Tanpa `--delete`.
Transfer artefak dilakukan terpisah agar perubahan source tidak menimpa hasil run.

## Berikutnya

Prioritas aktif berada di [ROADMAP](ROADMAP.md): tunggu laporan uji rekaman baru pengguna, lalu review bukti coverage/geometri/denah dan tentukan refinement atau capture tambahan. Usulan detail/texture, recovery generik, MapAnything dan splat adalah tahap terpisah. Skala relatif dan interior unseen belum terverifikasi; optimized splat/kernel belum diuji.
