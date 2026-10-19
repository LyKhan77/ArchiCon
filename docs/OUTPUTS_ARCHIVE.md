# Arsip eksperimen lama

**16 Oktober 2026.** Pengguna menyetujui arsip dingin. Eksperimen lama dipisahkan tanpa penghapusan; seluruh produk final tetap di `end_product/`. Latar belakang dan inventaris sebelum operasi: [analisis](OUTPUTS_CLEANUP_ANALYSIS.md). Bukti pelaksanaan berada di `outputs/documentation_audit/outputs-retention-2026-10-16/`.

```text
outputs/
  jobs/lantai3_part1_1016_1525/     Job rekaman baru, tetap lengkap
  .run-index/                     Registry asli, termasuk ID eksperimen lama
  .locks/, .dashboard/             Infrastruktur operasional
  dashboard_qa/, documentation_audit/
  archive/experiments_2026-10-16/
    flat/<nama_asli>/              Eksperimen, job kontrol, stage dan log datar
    jobs/<job_uji>/                archicon_cache_001, layout_cache_001, layout_smoke_001
    move-manifest.json             Mapping asal/tujuan, SHA tiap file, mode/mtime
    move-manifest.sha256           Checksum manifest
    move-journal.jsonl             Intent/done tiap rename dan verifikasi
    README.md                      Cara restore
```

| Host | Entri diarsipkan | File terverifikasi | Ukuran logis |
|---|---:|---:|---:|
| Laptop | 83 | 2.988 | 2,161 GiB |
| Server | 141 | 15.728 | 12,768 GiB |

Arsip kedua host berbeda isi/subset; tidak digabung atau dianggap duplikat. Pemindahan dilakukan dengan rename pada filesystem yang sama sehingga tidak menghemat kapasitas disk. Seluruh18versi produk final tersedia di kedua host setelah backup delapan versi yang sebelumnya hanya ada di laptop. Produk, job pengguna, registry, sumber video, model/runtime dan symlink menuju proyek/cache lama tetap dipertahankan.

## Penggunaan sehari-hari

Jalankan ArchiCon/pipeline seperti biasa untuk video baru. History dashboard hanya menampilkan job yang masih berada pada layout aktif. HTML/mesh/denah final versi lama tetap dapat dibuka langsung dari `end_product/`.

Untuk melihat laporan eksperimen, buka path pada `archive/.../flat/` atau `archive/.../jobs/`. Isi artefak, fingerprint, JSON dan entry registry historis tidak ditulis ulang. Karena path asal disimpan dalam checkpoint, **restore dahulu** sebelum memakai status/resume/reuse/refinement job lama atau exporter baseline `configs/room_tour.json` dan `configs/reuse_l2bp3.json`. Pipeline fresh membuat konfigurasi sendiri di job baru dan tidak membutuhkan restore.

## Verifikasi dan restore

Dari root proyek laptop:

```sh
rtk proxy .venv/bin/python scripts/archive_outputs.py verify
rtk proxy .venv/bin/python scripts/archive_outputs.py restore
```

`verify` membaca seluruh file dan mencocokkan SHA, ukuran, struktur folder, mode dan mtime dengan manifest. `restore` memeriksa seluruh daftar sebelum memindahkan apa pun, lalu mengembalikan tepat ke path asal. Ia menolak symlink, path keluar workspace, file yang berubah, collision/overwrite, filesystem berbeda, dan lock controller/worker yang sedang dipakai. Jangan menjalankan job saat melakukan restore. Restore seluruh kelompok diperlukan agar dependensi lintas-run tersedia kembali.

Untuk server, jalankan dari laptop:

```sh
rtk proxy ssh gpu-server 'cd /srv/archicon && .venv/bin/python scripts/archive_outputs.py verify'
rtk proxy ssh gpu-server 'cd /srv/archicon && .venv/bin/python scripts/archive_outputs.py restore'
```

Manifest terikat root dan host; jangan memakai manifest laptop untuk operasi server. Utilitas tidak memiliki fungsi hapus. Jika apply/restore terputus, jalankan kembali perintah yang sama: lokasi dan hash aktual diperiksa, entri yang sudah dipindah tidak ditimpa. Untuk mengarsipkan kembali setelah restore, gunakan `apply` dengan manifest yang sama. `prepare` hanya untuk membekukan proposal baru, bukan menjalankan ulang arsip yang sudah memiliki manifest.

Penghapusan di masa depan membutuhkan daftar konkret dan konfirmasi pengguna tersendiri. Persetujuan arsip ini tidak memberi izin menghapus arsip atau produk final.

Bukti: verifikasi pelaksanaan (`../outputs/documentation_audit/outputs-retention-2026-10-16/execution-verification.json`, lokal), cek job/HTTP/ZIP/repackaging (`../outputs/documentation_audit/outputs-retention-2026-10-16/active-job-verification.json`, lokal), manifest laptop (`../outputs/archive/experiments_2026-10-16/move-manifest.json`, lokal). Manifest server tetap pada host server; jangan menimpanya dengan salinan laptop.
