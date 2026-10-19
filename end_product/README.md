# Hasil akhir

Job baru: `<job_id>/vNNN/`. Setiap versi berisi `mesh/`, `denah/`, `room_tour/index.html`, `reports/`, `calibration.json`, `manifest.json` dan README. `latest.json` pada folder job menunjuk versi terbaru beserta status review/kualitas.

Buka `room_tour/index.html` langsung di browser. File HTML offline; mesh detail dan denah menjadi file pendamping. Review pending berarti belum dinilai manual; completed hanya menyatakan komputasi selesai. Dimensi tetap relatif sampai ada ukuran nyata.

Arsip lama tetap pada `part1/` dan `part2/`:

- PART1 (`part1/README.md`, lokal), immutable.
- Hasil fresh full002 (`part2/part2_full_002/v002/README.md`, lokal), needs_capture.
- Baseline dengan entrance (`part2/part2_cache_001/v003/README.md`, lokal), needs_capture.
- Smoke yang direview (`part2/part2_smoke_003/v003/README.md`, lokal), accepted_partial.

Detail stage/log/QA/review tetap pada [outputs](../outputs/README.md). Jangan mengubah file dalam versi finalized; manifest menyimpan checksum-nya. Pipeline membuat versi baru saat keputusan review diterapkan.
