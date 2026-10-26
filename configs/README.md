# Konfigurasi

Profil mesin aktif di-ignore. Checkout baru menyalin `paths.local.example.yaml` dan `paths.server.example.yaml` ke nama tanpa `.example`, lalu mengisi root/SSH/UUID sesuai mesin. Jangan menimpa profil workspace yang sudah berjalan. [Setup](../docs/SETUP.md).

| File | Peran |
|---|---|
| `paths.local.yaml` | Workspace laptop, review, dan target sinkronisasi |
| `paths.server.yaml` | Workspace server, satu GPU, resource budget, dan sumber eksternal |
| `capture_l2bp3.yaml` | Metadata input terverifikasi dan kebijakan decoding |
| `models.yaml` | Exact snapshot checkpoint yang ditemukan di cache server |
| `pilot.yaml` | Rentang dipertahankan setelah audit pilot; parameter depth masih rencana |
| `full_take.yaml` | Kondisi yang wajib dipenuhi sebelum pemrosesan full take |
| `pipeline.yaml` | Profil E2E default: adaptive 2/4 fps, sequential overlap32, S06G dan review manual |
| `pipeline.dense.yaml` | Profil E2E opsional: frame asli 4 fps, sequential overlap64, SIFT1600/8192; lebih lambat, bukan jaminan geometri lengkap |

Loader YAML, resource gate, dan unique run ID sudah diterapkan pada entrypoint yang
tersedia. Capture config digunakan audit; profile pilot/full take memuat rencana
tahap berikutnya dan belum menjadi orchestrator otomatis. Camera CLI masih memakai
argumen eksplisit, dengan default rentang yang sama seperti profile pilot.

Path relatif di-resolve terhadap project root profile terpilih. Kredensial tidak
ditaruh dalam konfigurasi. Checkpoint legacy dipin untuk reproduksi; pemilihan
checkpoint baru adalah eksperimen terpisah.

`reuse_l2bp3.json` dan `room_tour.json` adalah konfigurasi baseline historis. Sumber outputnya berada pada arsip dingin; [restore](../docs/OUTPUTS_ARCHIVE.md) diperlukan sebelum menjalankannya. Untuk video baru, orchestrator membuat konfigurasi capture/export sendiri di `outputs/jobs/<job>/config/`.
