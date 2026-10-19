# Analisis pemisahan output eksperimen lama

**Tanggal:**16 Oktober2026. **Status dokumen:**snapshot analisis sebelum operasi. Pengguna kemudian menyetujui arsip dingin tanpa penghapusan; hasil pelaksanaan dan cara restore dicatat pada [OUTPUTS_ARCHIVE](OUTPUTS_ARCHIVE.md). Angka/proposal di bawah mempertahankan keadaan saat audit, termasuk temuan backup sebelum disalin. Penghapusan tetap membutuhkan konfirmasi eksplisit untuk daftar konkret.

## Ringkasan

Pipeline E2E untuk job baru sudah terpisah. Telusur metadata job `lantai3_part1_1016_1525` tidak menemukan dependensi ke folder eksperimen lama. Job tidak memakai reuse-config dan seluruh15entry registry dalam bundlenya menunjuk ke stage job itu sendiri, dengan hash cocok. Referensi ke registry bersama tidak ditelusuri sebagai dependensi ke semua run: hanya entry milik job yang relevan. File konfigurasi mesin masih menyimpan alias legacy, tetapi jalur fresh menggunakan input/capture/job yang diberikan secara eksplisit. Pengguna melaporkan hasil overall OK; analisis ini tidak mengubah review pending menjadi keputusan kualitas formal.

| Inventaris | Laptop | Server |
|---|---:|---:|
| Ukuran logis outputs saat audit | 2.288 GiB | 14.726 GiB |
| Kandidat dipisahkan sebagai arsip, bukan dihapus | 83 entri / 2.161 GiB | 141 entri / 12.768 GiB |
| Entri datar eksperimen lama | 80 | 138 |
| Job uji nested lama | 3 | 3 |
| Metadata/text yang dibaca | 961 | 1554 |
| File metadata >16MiB dilewati / error baca | 0 / 0 | 0 / 0 |

Ukuran adalah byte logis, bukan jaminan kapasitas yang dapat dibebaskan. Hardlink/allocated blocks dicatat pada JSON/CSV. Memindahkan folder ke arsip pada disk yang sama merapikan struktur tetapi tidak mengurangi pemakaian disk. Tidak ada daftar hapus yang disetujui.

## Yang tetap dipertahankan

- Seluruh job pengguna: `outputs/jobs/lantai3_part1_1016_1525/`, termasuk kamera/depth/geometri, konfigurasi, review, QA dan log. Snapshot controller lokal completed/review pending; worker server complete. Tidak melakukan review visual atau mengubah keputusan kualitas pada analisis ini.
- Registry/lock/controller: `.run-index/`, `.locks/`, `.dashboard/` bila tersedia; `.gitkeep` dan README. Registry lama tidak dihapus karena mengikat identitas run dan diperlukan untuk restore.
- Bukti validasi `dashboard_qa/` dan `documentation_audit/`; report audit baru berada di dalam documentation_audit, tidak menambah namespace top-level baru.
- **Seluruh `end_product/`**:18versi paket di laptop dan10versi di server, termasuk PART1 dan semua versi lama. Sepuluh manifest yang ada pada kedua host cocok SHA-256. Standalone HTML/GLB/denah final tidak dipindahkan.
- Semua raw video, vendor, environment, model/cache dan proyek lama. Tidak ada pruning bobot atau input sumber.

## Kelompok eksperimen lama

| Kelompok | Laptop | Server | Perlakuan yang diusulkan |
|---|---:|---:|---|
| Pilot/audit/kamera/depth/filter/fusion/render/denah awal, `l2bp3_*` + `legacy_l2bp3_001` + preview | 48entri /1,715GiB | 51entri /6,772GiB | Arsip kelompok utuh; jangan menghapus depth/kamera satu per satu |
| Kontrol job E2E lama `part2_*` | 9entri /0,005GiB | 9entri /0,007GiB | Arsip dengan stage, review, log dan hasil perbandingan |
| Stage E2E datar `part2_*_S*` | 20entri /0,393GiB | 40entri /5,475GiB | Arsip bersama job asalnya |
| Preflight lama | 3entri /<0,001GiB | 7entri /<0,001GiB | Arsip provenance; bukan penghematan disk berarti |
| Console logs datar | 0 | 31file /0,001GiB | Arsip bersama run asal; bukan prioritas penghapusan |
| Job uji nested | archicon_cache_001,layout_cache_001,layout_smoke_001 | Sama | Arsip bila dianggap closed; hasil final tetap dipertahankan |

CSV lengkap per path: laptop (`../outputs/documentation_audit/outputs-retention-2026-10-16/local-move-proposal.csv`, lokal), server (`../outputs/documentation_audit/outputs-retention-2026-10-16/server-move-proposal.csv`, lokal). Proposal mesin: move-proposal.json (`../outputs/documentation_audit/outputs-retention-2026-10-16/move-proposal.json`, lokal). Semua baris adalah kandidat, bukan izin hapus.

## Dependensi yang membuat pemindahan manual berisiko

1. `configs/reuse_l2bp3.json` masih mereferensikan audit, kamera terkoreksi, depth/full/entrance, surface/render dan `l2bp3_room_tour_005`. Jalur reuse baseline ini akan membutuhkan restore jika sumbernya diarsipkan dingin.
2. `configs/room_tour.json` memakai `l2bp3_full_delivery_001`, manifest render dan kamera lama. Ini exporter baseline historis; pipeline fresh membuat konfigurasi exporter sendiri di job.
3. Bundle dan review `part2_cache_001`, `archicon_cache_001` serta `layout_cache_001` terikat path dan SHA bukti lama. Pindah file tidak mengubah hash isinya, tetapi lookup review/resume lama gagal sampai path asal dipulihkan.
4. `part2_full_002` memakai audit fresh `part2_full_001_S02_audit_*`; kandidat refinement smoke memakai kamera/depth baseline smoke. Memisahkan stage satu per satu akan memutus kelompok ini.
5. Resolver saat ini hanya mengenali output datar lama dan registry ke `outputs/jobs/JOB/stages/STAGE/RUN_ID`. Ia menolak symlink output. Karena itu, shortcut symlink bukan solusi kompatibilitas yang valid.
6. Sejumlah manifest memuat path absolut server, relative paths serta upstream hashes. Mengedit manifest lama akan merusak fingerprint; jangan melakukan search/replace di artefak immutable.

Graph konservatif laptop:33entri/0,729GiB tersambung ke job/produk/config;21entri/1,032GiB menjadi dependensi historis lain;17entri/0,347GiB tidak ditemukan incoming operational reference. Server:57/10,317GiB,21/1,535GiB,51/0,396GiB. Referensi dokumentasi/provenance tidak berarti semua file diperlukan untuk membuka HTML final. Sebaliknya, tidak ada incoming reference bukan bukti aman dihapus; dataset, diagnostik dan reproducibility masih memiliki nilai.

## Temuan backup dan batas kepemilikan

Paket job baru `end_product/lantai3_part1_1016_1525/v001` baru ada di laptop. Sebelum pemisahan fisik, buat salinan server dan verifikasi setiap artefak terhadap manifest. Tujuh versi historis lain juga belum ada di server; tetap dilindungi di laptop. Folder bernama sama pada laptop/server dapat berisi subset berbeda; jangan menganggap laptop adalah duplikat yang aman dihapus hanya dari nama folder atau ukuran. Daftar versi yang hanya ada di laptop:

- `end_product/lantai3_part1_1016_1525/v001`
- `end_product/part2/part2_cache_001/v001`
- `end_product/part2/part2_full_002/v001`
- `end_product/part2/part2_metadata_001/v001`
- `end_product/part2/part2_smoke_003/v001`
- `end_product/part2/part2_smoke_003/v002`
- `end_product/part2/part2_smoke_003_r1c1/v001`
- `end_product/part2/part2_smoke_003_r1c2/v001`

Pada server, `raw_videos/lantai2_B_part3.MOV` menunjuk sumber video proyek lama; `data/legacy/l2bp3_stream` dan `l2bp3_viewer` adalah symlink ke baseline proyek lama; `models/huggingface` menunjuk shared cache. Audit tidak mengikuti/mengukur target symlink tersebut dan tidak memberi izin memindahkan atau menghapusnya. `data/legacy` lokal hanya berisi .gitkeep.

Snapshot proses: tidak ditemukan controller/worker rekonstruksi aktif di laptop/server; backend ArchiCon lokal tetap hidup. Ini hanya snapshot audit, bukan izin menjalankan move tanpa mengecek ulang.

## Rekomendasi pemisahan

Pilihan yang disarankan adalah **arsip dingin hasil eksperimen yang closed**, tanpa penghapusan. Job baru tetap berada pada layout aktif dan seluruh produk final tetap pada tempatnya. Kelompok lama dapat dipulihkan ke path persis semula bila suatu saat diperlukan untuk resume/reuse/refinement.

```text
outputs/
  jobs/lantai3_part1_1016_1525/      Job pengguna aktif/terbaru
  .run-index/, .locks/, .dashboard/ Infrastruktur yang tetap
  dashboard_qa/, documentation_audit/ Bukti validasi
  archive/experiments_2026-10-16/
    flat/<nama_asli>/               Seluruh eksperimen/job/stage datar
    jobs/<job_uji>/                 Tiga job uji nested
    move-manifest.json              Mapping origin/destination + hash
    move-journal.jsonl              Progress move dan rollback
    README.md                       Cara melihat dan restore
```

Dengan arsip dingin, eksperimen lama sengaja keluar dari history dashboard dan tidak bisa langsung resume/reuse sebelum restore. Entry registry historis tetap disimpan/reserved; targetnya tersedia kembali setelah restore. Model/cache, input video, user job, dan produk final tidak ikut arsip. Jika pengguna membutuhkan resume/reuse lama tanpa restore, perlu implementasi archive-aware lookup dan verifikasi semua pembaca path; bukan sekadar mv.

## Urutan setelah kebijakan arsip dipilih

1. Backup paket final baru ke server, cek manifest/byte hashes; lindungi18versi lokal dan10versi server yang sudah ada.
2. Bekukan daftar origin/destination eksplisit dari CSV. Jangan gunakan wildcard l2bp3*/part2* untuk operasi move/delete.
3. Siapkan utilitas dry-run/apply/restore dengan validasi containment/no-symlink/no-overwrite, per-file SHA, journal atomik dan rollback terarah. Tidak ada fungsi delete.
4. Cek ulang proses/lock milik workspace; jalankan move sebagai rename pada filesystem yang sama. Jangan menyentuh source lama atau shared cache.
5. Verifikasi semua file yang dipindah dan semua produk/job pengguna tetap cocok hash. Uji status/dashboard/repackage dari job baru tanpa inference GPU. Lakukan smoke restore/rollback pada fixture temporer terlebih dahulu.
6. Perbarui tautan dokumentasi artefak historis serta petunjuk restore; tandai configs reuse/export baseline sebagai memerlukan restore dalam mode arsip dingin.
7. Jangan menghapus arsip. Jika nanti dibutuhkan, ajukan daftar path, ukuran, alasan, dependensi dan backup yang spesifik; tunggu konfirmasi pengguna sebelum penghapusan.

## Bukti audit

- Inventaris laptop (`../outputs/documentation_audit/outputs-retention-2026-10-16/local-inventory.json`, lokal)
- Inventaris server (`../outputs/documentation_audit/outputs-retention-2026-10-16/server-inventory.json`, lokal)
- Ringkasan (`../outputs/documentation_audit/outputs-retention-2026-10-16/summary.json`, lokal)
- Scanner read-only (`../outputs/documentation_audit/outputs-retention-2026-10-16/read_only_audit.py`, lokal)

Tidak ada model, data biner mesh/depth, atau visual interior yang dikirim ke layanan review. Analisis memakai stat file, metadata/text≤16MiB dan hash manifest, bukan inference atau review kualitas visual. Tidak ada file data yang dipindah atau dihapus pada tahap ini.
