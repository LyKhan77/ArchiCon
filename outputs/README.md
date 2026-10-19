# Output internal pipeline

Job baru disimpan di `jobs/<job_id>/`:

```text
job.json, job-status.json, worker-status.json, bundle.json
config/                       # Machine, capture, mask, exporter config
logs/                         # Worker, launcher, telemetry log stage, transfer list
stages/<stage>/<run_id>/       # Attempt numerik immutable
qa/<delivery_version>/        # Screenshot dan laporan browser
review/round_NN/request_NNN/   # Snapshot bukti, schema, hasil review manual
stages.json, events.jsonl      # Checkpoint dan event log
```

`.run-index/` menghubungkan run-ID CLI dengan lokasi stage; ikut diverifikasi saat transfer. `.locks/` berisi lock GPU bersama.

Eksperimen lama dipisahkan ke `archive/experiments_2026-10-16/{flat,jobs}/`, tanpa penghapusan. Job rekaman baru tetap pada `jobs/`; seluruh produk final tetap pada `end_product/`. Manifest dan registry lama tidak diedit: restore diperlukan sebelum status/resume/reuse/refinement eksperimen lama. Lihat [panduan arsip dan restore](../docs/OUTPUTS_ARCHIVE.md). Memindahkan pada disk yang sama tidak mengurangi pemakaian kapasitas.

Hasil untuk dibagikan ada di [end_product](../end_product/README.md). Alur penggunaan: [WORKFLOW](../WORKFLOW.md).
