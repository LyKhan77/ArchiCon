# ROADMAP — ArchiCon

**Status:** diperbarui 26 Oktober 2026. Pipeline E2E dan adaptive sampling tersedia. S06G mendukung kandidat gabungan bila shared-camera alignment lolos; pada hasil lantai3_part2 saat ini world belum tersambung. Prioritas aktif: recovery konektivitas kamera, bukan peningkatan tampilan HTML. Review kualitas tetap manual.

Rujukan: [PRD](PRD.md), [TECHSTACK](TECHSTACK.md), [ARCHICON](ARCHICON.md), [WORKFLOW](../WORKFLOW.md), [PROGRESS](PROGRESS.md). Rencana riset P0–P6 lama disimpan di arsip roadmap (`archive/ROADMAP-research-2026-10-14.md`, lokal); tidak menjadi daftar fitur yang sudah tersedia.

## 1. Jalur produk saat ini

Video → audit orientasi/PTS/keyframe → PyCOLMAP CPU → kamera/komponen terverifikasi → DA3 pose-conditioned depth → mask/filter/CPU TSDF → render dan denah → mesh viewer + HTML offline → browser QA → review manual.

Laptop menjalankan ArchiCon/controller, SSH/rsync, packaging dan QA. Server menjalankan audit, kamera, inference dan geometri dengan runtime terpisah. Hanya satu UUID GPU pada profil aktif yang diizinkan; semua guard RAM/disk/GPU tetap berlaku. MapAnything, Gaussian Splatting dan semantic architectural floorplan belum diterapkan.

## 2. Milestone yang selesai

| Milestone | Hasil yang tersedia | Batas bukti |
|---|---|---|
| Workspace dan resource | Profil lokal/server, cache lama read-only, runtime audit/inference/geometry, guard dan provenance | Runtime inference masih mewarisi package sistem/user; ketersediaan resource bukan reservasi |
| Audit video | Rotasi, durasi/PTS, frame hashes, sharpness/exposure, contact sheet dan all-frame accounting | Keyframe/coverage bukan bukti seluruh interior terlihat |
| Kamera | Pilot/validation, full take, uniform/adaptive sampling, matching, undistortion dan karantina pose ekstrem | Recovery PnP bridge antar-world belum tersedia |
| Global architecture | S06G shared-camera Sim(3), fit/validation/orientation/cycle gates, GLB/denah/HTML kandidat bila lolos | World tanpa shared frame cukup tetap fallback; belum global TSDF atau full-building coverage |
| Depth dan mesh | DA3-LARGE lokal, window bounded, filtering, mask orang/anotasi, Open3D CPU TSDF, render triangle | Mask kaca tidak lengkap, holes/outlier mungkin ada, skala relatif |
| Denah dan viewer | Draft SVG/PNG, mesh viewer ringan, orbit/roof clipping/viewpoints/playback, HTML offline, kalibrasi JSON per-komponen | Bukan denah arsitektur semantik atau panorama 360; acuan nyata belum ada |
| Orchestrator | run/status/resume, worker detached, fingerprint/hash, transfer bundle, produk versi baru | Completed berarti komputasi selesai, bukan kualitas diterima |
| Review/refinement | Handoff JSON manual, schema/evidence checks, preset S06/S07 maksimal2ronde×2kandidat | Tidak memanggil agent/LLM otomatis; tidak melakukan completion geometri |
| Layout job | config/stages/logs/qa/review nested, run registry, kompatibilitas hasil lama | Manifest/registry lama dipertahankan; resume/reuse perlu restore arsip |
| ArchiCon | Input/upload/preview, history, progress/log, resume, hasil/ZIP, dark mode DESIGN.md | Backend localhost tetap dibutuhkan untuk UI; satu controller dashboard aktif |
| Desain room tour | Workspace arsitektur/interior selaras ArchiCon, komponen dengan thumbnail, font embedded, denah/kalibrasi dan mobile | v002 adalah pembaruan presentasi; mesh dan embedded data tetap identik dengan v001 |
| Arsip eksperimen | 83entri laptop/141server dipisahkan, SHA per file, journal dan restore;18paket final tersedia di kedua host | Arsip dingin membutuhkan restore sebelum reuse/resume lama; tidak menghemat kapasitas disk |

## 3. Bukti pengujian acuan

| Run / paket | Jenis uji | Hasil |
|---|---|---|
| part2_full_001 | Cold-run seluruh video, tanpa reuse rekonstruksi | 18stage,280/302frame,5world, QA HTML offline/mobile lulus |
| part2_full_002/v002 | Audit fresh dipakai ulang; kamera/depth/mesh dihitung ulang dengan auto/exhaustive | 283/302frame,19gap,satu world utama, review needs_capture |
| part2_cache_001/v003 dan PART1 | Arsip baseline dengan entrance-local terpisah | 286/302frame,16gap; world utama/entrance belum aligned, needs_capture |
| part2_smoke_003 | Fresh30s dan dua kandidat refinementS06 nyata | 60/60; kedua kandidat dibandingkan, baseline dipertahankan, accepted_partial |
| layout_smoke_001/v001 | Fresh30s di layout nested | 6stage baru,60/60, QA offline/mobile lulus; review pending/unassessed |
| archicon_cache_001/v001 | Packaging video acuan melalui HTTP resume | 286/302,dua komponen, QA/ZIP lulus; cache-only, review pending/unassessed |

Jumlah tes pada regresi dashboard:246lokal (242lulus,4dilewati),74server pipeline/dashboard. Bukti: verifikasi dashboard (`../outputs/dashboard_qa/verification.json`, lokal), verifikasi layout (`../outputs/archive/experiments_2026-10-16/jobs/layout_smoke_001/layout-verification.json`, lokal), verifikasi full002 (`../outputs/archive/experiments_2026-10-16/flat/part2_full_002/final-verification.json`, lokal). Ini snapshot pengujian sebelum hasil rekaman baru pengguna, bukan status live semua job.

## 4. Struktur aktual

```text
README.md, WORKFLOW.md, DESIGN.md       Panduan root dan desain
configs/, environments/               Profil dan pin runtime
src/room_reconstruction/, scripts/    Implementasi dan entrypoint
web/dashboard/, web/tour/              Dashboard lokal dan viewer offline
tests/, schemas/                       Tes dan kontrak review
docs/                                 Dokumentasi utama, rencana dan arsip
raw_videos/                           Input pengguna dan upload browser
outputs/jobs/<job>/                    Config, stages, logs, qa, review
outputs/.run-index/                    Registry run-ID immutable
outputs/.dashboard/                    Metadata controller/ZIP temporer
outputs/archive/experiments_2026-10-16/ Eksperimen closed + manifest/journal restore
end_product/<job>/vNNN/                Mesh, denah, room_tour, laporan, manifest
```

Root PRD/ROADMAP/PROGRESS adalah pointer kompatibilitas. Hasil eksperimen lama dipindah utuh ke arsip; end_product/part1 dan end_product/part2 tetap pada tempatnya. Stack, file penanggung jawab dan runtime dijelaskan di [TECHSTACK](TECHSTACK.md); [restore arsip](OUTPUTS_ARCHIVE.md).

## 5. Prioritas setelah uji video baru

- [x] Implementasikan S06G dan jalankan reuse hasil `lantai3_part2` tanpa inference baru. Uji nyata menghasilkan `no_alignment`: 17 world delivery tidak mempunyai shared frame-ID. Bukti lengkap di laporan delivery baru; bukan penerimaan kualitas.
- [x] Bandingkan kamera 4 fps penuh, overlap 64, resolusi SIFT 1600/8192 fitur: registrasi naik 73,7% → 82,5%, largest-map share 15,7% → 28,7%. Kandidat dense selesai tanpa mengubah produk lama.
- [x] Gabungkan dua world dari kandidat dense melalui 20 shared cameras dengan 6 kamera validasi. `lantai3_aligned_1026/v001` membawa architecture GLB/denah/HTML, seluruh 824 viewpoint dan 10 fallback; QA browser lulus. Belum seluruh bangunan, artefak depth masih ada.
- [ ] Bangun korespondensi gambar/PnP antar-world bila kamera tetap terpecah; validasi residual serta geometrinya sebelum gabungan.
- [ ] Global TSDF/optimasi pose dan deduplikasi permukaan setelah transform lintas-world berhasil, baru fokus menutup gap yang punya bukti.

- [x] Dashboard dan dokumentasi cara run tersedia.
- [x] Regresi stack acuan, HTTP packaging, upload/duplikasi ID dan desktop/mobile selesai.
- [x] Terima laporan `lantai3_part1_1016_1525`: pengguna menyatakan selesai, overall OK. Review formal tetap pending/unassessed.
- [x] Pisahkan eksperimen lama tanpa penghapusan, backup versi final yang belum tersalin, dan verifikasi integritas/dashboard/packaging job baru.
- [x] Selaraskan desain HTML room tour dengan dashboard ArchiCon/ DESIGN.md; v002 tersedia, fungsi viewer/geometri dipertahankan, QA offline/mobile/HTTP/ZIP lulus. [Panduan](ROOM_TOUR.md).
- [ ] Setelah hasil diberikan, review semua komponen, source/render, seluruh timeline, coverage/gap dan denah.
- [ ] Tentukan keputusan: menerima hasil parsial, meminta refinement terbatas, atau capture tambahan dengan lokasi/timestamp yang jelas.
- [ ] Terapkan JSON review jika pengguna menghendaki, lalu bandingkan kandidat dari input identik dan buat versi hasil baru.

Tidak menjalankan ulang inference besar hanya untuk mengubah dokumentasi atau tanpa temuan yang membenarkan. Jangan memakai reuse-config video acuan pada rekaman baru. Status accepted_partial/needs_capture tidak boleh disamakan dengan full-room reconstruction complete.

## 6. Setelah review rekaman independen

| Prioritas | Pekerjaan | Gate / bukti sebelum selesai |
|---|---|---|
| 1 | Capture area yang tidak terlihat atau sambungan lemah | View dari dalam ruangan, overlap translasi, dan perbandingan coverage hasil baru |
| 2 | Kalibrasi ukuran nyata | Satu referensi terukur dan beberapa jarak validasi independen; laporan error ukuran |
| 3 | Denah semantik | Review dinding/pintu/furnitur dan batas ruangan; jangan menutup polygon dari gap tanpa bukti |
| 4 | Detail/resolusi/tekstur | Benchmark bounded dengan kamera, frame dan viewpoint pembanding yang sama |
| 5 | Recovery kamera generik / submap alignment | PnP/korespondensi overlap, residual/transform tervalidasi; world gagal tetap terpisah |
| 6 | Diagnostik operasional tambahan | Reconciliation worker stale dan visual failure package ketika semua komponen gagal |

## 7. Riset opsional yang ditunda

MapAnything initializer/conditioning, checkpoint DA3 metric/nested lain, DSINE, optimized scene/splat dan DN-Splatter/AGS-Mesh bukan dependency MVP sekarang. Uji hanya jika kegagalan atau kebutuhan kualitas membenarkan; catat model/revision/lisensi dan bandingkan A/B dengan budget/input yang sama. Kernel Blackwell forward/backward menjadi gate jika training Gaussian Splatting akan ditambahkan.

Multi-take/multi-floor, kamera360, collision/walk, CAD/DXF/PDF, layanan multiuser dan deployment publik juga belum diterapkan. Jangan mengubah UI, backend atau pipeline aktif berdasarkan rencana arsip tanpa tugas baru yang jelas.
