# ArchiCon

**Diperbarui:** 16 Oktober 2026. Pengguna melaporkan job baru selesai, overall OK; review formal tetap pending. History kini berisi job pengguna yang masih aktif pada layout, sedangkan eksperimen lama berada pada [arsip dingin](OUTPUTS_ARCHIVE.md). Produk final lama tetap dapat dibuka dari end_product/.

Dashboard HTML lokal untuk pipeline video → mesh, denah dan room tour. Nama dan dark mode mengikuti permintaan pengguna; bentuk siku, IBM Plex Sans lokal, garis tipis dan aksen IBM Blue mengikuti [DESIGN.md](../DESIGN.md) dengan adaptasi Carbon Gray100. Tidak memakai framework web tambahan atau CDN. Stack/runtime: [TECHSTACK.md](TECHSTACK.md). Status prioritas: [ROADMAP](ROADMAP.md).

## Jalankan

Dari root proyek di laptop:

```bash
rtk proxy .venv/bin/python scripts/serve_dashboard.py
```

Buka http://127.0.0.1:8765. Gunakan `--port 8766` jika port default dipakai aplikasi lain. Backend hanya mendengarkan localhost. Satu proses backend untuk workspace ini; semua tab browser memakai backend tersebut.

1. Pilih video dari perangkat atau pilih video yang sudah ada di raw_videos. Preview bergantung dukungan codec browser; pipeline tetap membaca video menggunakan decoder server.
2. Isi ID rekonstruksi baru: maksimal32karakter huruf/angka/underscore/hyphen. ID lama tidak ditimpa. Pilih scope `Full traversal`, `Single room`, atau `Connected area`; scope selain full traversal meminta nama area dan menerima timestamp opsional. Upload dibatasi20GiB dan ditulis bertahap ke raw_videos/uploads/UPLOAD/source.EXT; hanya upload lengkap yang muncul sebagai input.
3. Klik Mulai rekonstruksi. Job baru selalu fresh, tanpa reuse-config video acuan. Controller melakukan registrasi/checksum, sync kode/video, menjalankan worker server, mengambil artefak, membangun HTML dan menjalankan browser QA.
4. Tab Progress menunjukkan fase transfer/download/packaging/QA serta attempt stage numerik dari server. Tidak ada persentase komputasi atau ETA buatan. Persentase upload adalah jumlah byte browser → laptop, terpisah dari transfer SSH laptop → server. Preview audit/render muncul setelah file tersedia. Snapshot dapat tertinggal beberapa detik; gangguan SSH ditampilkan sebagai status stale.
5. Tab Hasil akhir menyediakan room tour dalam iframe atau tab baru, denah, file mesh/laporan, sesi review global dan paketZIP. Produk tetap berada di end_product/JOB/vNNN, tanpa menyalin data depth mentah ke browser.

Room tour memakai workspace arsitektur/interior dark mode yang selaras dengan dashboard: thumbnail komponen, kontrol potongan model, canvas 3D, lembar denah terang dan pengukuran draft. Font juga embedded sehingga HTML tetap offline. Untuk job `lantai3_part1_1016_1525`, hasil terbaru adalah v002; refresh halaman jika sebelumnya masih membuka v001. [Panduan kontrol room tour](ROOM_TOUR.md).

## Pemulihan dan review

Pengunci controller dipegang proses detached, sehingga reload/penutupan browser atau restart backend tidak membuka slot job kedua. Pembatas ini mengatur controller dari dashboard; resource/GPU guard pipeline tetap memeriksa proses lain di server. Dashboard tidak mengantrekan job atau menurunkan batas resource.

Saat controller berhenti/gagal atau menunggu resource, klik Lanjutkan proses setelah penyebab selesai. Resume memanfaatkan artefak yang lolos pemeriksaan pipeline. Jika worker server masih hidup, controller mengambil kembali monitoring. Tidak ada tombol menghentikan proses server lain.

Komputasi selesai berbeda dari kualitas diterima. Pada Tab Hasil akhir, sesi review global menyediakan Accept, Need review, dan Reject/perlu rekaman ulang. Reject meminta alasan dan dipetakan internal ke `needs_capture`; Accept dengan gap menjadi `accepted_partial`. Review tetap lokal dan manual, tanpa agent/LLM. JSON/CLI `apply-review` tetap tersedia untuk review detail. Denah masih draft dan berskala relatif sampai ukuran nyata dikalibrasi dalam room tour; processed-frame coverage tidak membuktikan seluruh permukaan ruangan lengkap. Scope hanya mengatur label/provenance dan pilihan frame, bukan klaim kelengkapan.

## Implementasi

- web/dashboard/: halaman, CSS, JS, font dan lisensi lokal.
- scripts/serve_dashboard.py: serverHTTP localhost.
- scripts/dashboard_controller.py: wrapper resume dengan kernel lock yang diwariskan.
- src/room_reconstruction/dashboard_service.py: upload streaming, registrasi/controller, file publik dan ZIP.
- src/room_reconstruction/dashboard_http.py: HTTP/API, byte-range video, pemeriksaan Host/origin/token.
- src/room_reconstruction/dashboard_snapshot.py: ringkasan stage/log bounded di server.
- outputs/.dashboard/: metadata controller dan ZIP temporer; ZIP dibuang setelah download.
- outputs/jobs/JOB/controller-progress.json: fase controller; dashboard-snapshot.json: cache progress server.
- outputs/jobs/JOB/logs/dashboard-controller.log dan qa/live/: log serta preview proses.

Input model dan kredensial SSH tetap pada lingkungan yang sudah dikonfigurasi; tidak ada pengisian credential di browser. Koneksi yang dipakai backend adalah SSH laptop yang sama dengan CLI pipeline.

## Bukti yang tersedia

QA desktop/mobile dashboard historis lulus;246tes lokal (242lulus,4dilewati) dan74tes pipeline/dashboard server. Integrasi `archicon_cache_001` membuktikan packaging HTTP/transport/QA dari cache video acuan, bukan inference rekaman baru. Fresh numerical layout regression terpisah: `layout_smoke_001`,60/60frame. Bukti historis: verification.json (`../outputs/dashboard_qa/verification.json`, lokal). Pembaruan room tour v002 lolos258tes lokal (254lulus,4dilewati),7tes JS kalibrasi serta QA offline seluruh komponen dan integrasi HTTP/ZIP. Pengguna melaporkan video baru overall OK; review kualitas formal tetap pending.
