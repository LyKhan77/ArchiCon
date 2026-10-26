# ArchiCon — Video → Mesh, Denah dan Room Tour

Pipeline untuk video iPhone/ruangan: kamera COLMAP → DA3 depth → filtering/TSDF → mesh detail, denah draft dan satu HTML offline. Video penuh acuan dan smoke30s telah diuji. Semua keyframe dicatat sebagai processed atau gap; kelengkapan frame tidak menjamin seluruh interior terlihat.

Review tetap **manual**. Script menyiapkan output dan QA lalu berhenti; tidak memanggil agent/LLM atau layanan review otomatis. Dimensi memakai unit relatif sampai ada pengukuran nyata.

Profil CLI opsional `--pipeline-config configs/pipeline.dense.yaml` memakai frame asli 4 fps dan matching lebih rapat. S06G dapat menghasilkan kandidat mesh/denah/HTML gabungan untuk area yang alignment kameranya lolos validasi; area lain tetap fallback. Ini belum jaminan rekonstruksi seluruh bangunan. [Alur dan batas](WORKFLOW.md).

Checkpoint source: [CHANGELOG.md](CHANGELOG.md). Untuk checkout baru, ikuti [SETUP](docs/SETUP.md); profil mesin aktif, media/model, hasil rekonstruksi dan rencana kerja tidak disertakan Git. Panduan kontribusi dan batas operasi: [AGENTS.md](AGENTS.md).

## Dashboard web

```bash
rtk proxy .venv/bin/python scripts/serve_dashboard.py
```

Buka **http://127.0.0.1:8765**. ArchiCon memakai dark mode sesuai [DESIGN.md](DESIGN.md): pilih/upload video, beri ID baru, lalu Mulai rekonstruksi. Progress menampilkan stage nyata, log, contact sheet/render setelah tersedia, dan coverage setelah dihitung. Hasil berisi room tour, denah, mesh, laporan, input review manual dan unduhan ZIP.

Satu controller dashboard aktif pada satu waktu. Browser dapat ditutup tanpa menghentikan controller; jika backend ditutup, jalankan lagi command yang sama untuk monitoring. Worker server terpisah tetap hidup. Jika controller gagal atau menunggu resource, periksa log lalu klik **Lanjutkan proses**. Review tidak dijalankan otomatis dan tidak dikirim ke layanan model. Panduan: [docs/ARCHICON.md](docs/ARCHICON.md).

## Menjalankan video baru

Dari root proyek pada laptop ini, simpan video yang sudah selesai disalin, misalnya `raw_videos/ruang_rapat/source.MOV`:

```bash
rtk proxy .venv/bin/python scripts/run_pipeline.py run \
  --input raw_videos/ruang_rapat/source.MOV \
  --job-id ruang_rapat_001 \
  --review-agent manual
```

ID job harus baru,1–32huruf/angka/underscore/hyphen. Jangan memakai reuse-config video acuan untuk rekaman lain. Script memverifikasi SHA input lokal/server dan memakai konfigurasi mesin yang sudah terpasang.

```bash
rtk proxy .venv/bin/python scripts/run_pipeline.py status --job-id ruang_rapat_001
rtk proxy .venv/bin/python scripts/run_pipeline.py resume --job-id ruang_rapat_001
```

`--detach` pada run melepas controller setelah worker server dimulai. Gunakan resume untuk mengambil hasil dan membuat HTML/QA saat laptop kembali tersedia. Worker server tetap hidup saat koneksi laptop putus. Resource wait tidak menurunkan batas otomatis.

## Struktur folder

```text
README.md, WORKFLOW.md          Panduan pengguna
src/, scripts/, configs/       Kode, entrypoint dan profil
web/, tests/, schemas/         Viewer, tes dan kontrak review
docs/                          PRD, roadmap, progress dan catatan pengembangan
raw_videos/<capture>/source.MOV Input video milik pengguna
outputs/jobs/<job_id>/
  job.json, job-status.json     Identitas dan status controller
  worker-status.json           Status proses server
  config/                      Machine/capture/mask/export config
  stages/<stage>/<run_id>/     Attempt numerik immutable
  logs/                        Worker, stage dan transfer logs
  qa/<version>/                Screenshot dan laporan browser
  review/                      Input/schema/keputusan review manual
end_product/<job_id>/
  latest.json                  Pointer ke versi terbaru + status kualitas
  v001/
    mesh/, denah/              GLB detail dan SVG/PNG draft
    room_tour/index.html       Satu file HTML offline
    reports/                   Coverage, kegagalan komponen dan review
    calibration.json           Default relatif; import/export draft pada HTML
    manifest.json, README.md   Checksum dan petunjuk paket
```

Eksperimen closed berada di `outputs/archive/experiments_2026-10-16/`; restore diperlukan sebelum resume/reuse lama. Produk `end_product/part1/` dan `part2/` tetap utuh pada workspace yang memilikinya. Root PRD/ROADMAP/PROGRESS adalah pointer kompatibilitas; dokumen utama berada di docs. `.run-index/` menghubungkan run-ID dengan stage bersarang, dan ikut diverifikasi saat transfer.

## Review hasil

Buka `end_product/<job_id>/vNNN/room_tour/index.html` langsung di browser. Periksa status review/kualitas pada manifest; completed hanya berarti komputasi selesai.

```bash
rtk proxy .venv/bin/python scripts/run_pipeline.py prepare-review --job-id ruang_rapat_001
```

Command mencetak lokasi input review. Setelah review manual menghasilkan JSON sesuai schema:

```bash
rtk proxy .venv/bin/python scripts/run_pipeline.py apply-review \
  --job-id ruang_rapat_001 \
  --review-file outputs/jobs/ruang_rapat_001/review/manual.json
```

Keputusan ditambahkan dalam versi baru tanpa mengubah byte produk yang direview. Refinement hanya preset S06/S07 tervalidasi, maksimal2ronde×2kandidat, dan berhenti lagi untuk review manual. Pipeline tidak mengarang area yang tidak terekam.

## Device dan resource

Laptop: registrasi, transfer, monitoring, HTML build dan browser QA. Server: audit, kamera, depth, fusion/render/export. Profil ada di `configs/paths.local.yaml`, `configs/paths.server.yaml` dan `configs/pipeline.yaml`.

Hanya GPU0 yang dipilih melalui UUID. Gate saat ini: RAM available≥8GiB, disk free≥50GiB, GPU free≥14GiB dan idle; CPU4thread/2worker. Runtime/model server terpisah dan sudah dipin. Default matcher auto memakai exhaustive hingga400keyframe, sequential di atasnya. Ini bukan jaminan setiap rekaman dapat direkonstruksi.

## Dokumentasi dan arsip

- [WORKFLOW](WORKFLOW.md): kontrak stage, resource, review dan batas kualitas.
- [Dokumentasi](docs/README.md): PRD, roadmap, progress.
- [Tech stack](docs/TECHSTACK.md): arsitektur, runtime/versi, model, resource dan format hasil aktual.
- [Room tour](docs/ROOM_TOUR.md): workspace arsitektur/interior ArchiCon, HTML offline dan kalibrasi; hasil v002 (`end_product/lantai3_part1_1016_1525/v002/room_tour/index.html`, lokal).
- [Catatan developer](docs/DEVELOPMENT.md): runtime, eksperimen dan command stage historis.
- [Output internal](outputs/README.md) dan [hasil akhir](end_product/README.md).
- Hasil fresh acuan (`end_product/part2/part2_full_002/v002/README.md`, lokal):283/302frame, needs_capture.
- Arsip baseline dengan entrance (`end_product/part2/part2_cache_001/v003/README.md`, lokal):286/302frame, needs_capture.

Pengguna melaporkan rekaman independen pertama selesai, overall OK; review formal tetap pending. Ukuran meter terverifikasi, pemulihan kamera generik dan denah arsitektur semantik masih pekerjaan tersisa. Geometri yang tidak pernah terlihat membutuhkan capture tambahan.
