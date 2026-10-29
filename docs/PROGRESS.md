# Progres eksekusi — 26 Oktober 2026

## Kamera pantry–workspace IT — 26 Oktober 2026

Video `lantai2_part4.MOV` diaudit pada 4 FPS: 555 frame asli dari rekaman sekitar 139 detik. CPU exhaustive matching mencakup 153.735 pasangan. Baseline mengembalikan satu world dengan 553/555 frame, tetapi pemeriksaan native COLMAP menemukan lonjakan lintasan sekitar 20, 62 dan 97,8 detik meski residual/calibration gate lulus. Frame sumber sekitar sambungan pertama tetap berurutan dan overlap; coverage tinggi saja tidak membuktikan tata ruang benar.

Eksperimen kalibrasi membaca cache baseline tanpa perubahan. Probe 13 pasangan mendukung pengujian lanjut dengan intrinsics hasil SfM; 78.506 pasangan cached diverifikasi ulang memakai dua worker satu thread sesuai profil, lalu incremental mapping dijalankan dengan empat thread dan ambang mapper/triangulasi semula. Tidak ada interpolasi frame, smoothing pose atau inference GPU.

| Diagnostik | Baseline | Kandidat kalibrasi |
|---|---:|---:|
| Observasi terdaftar | 553/555 | 553/555 |
| World diterima oleh gate kamera | 1 | 1 |
| Pantry / koridor / IT / kembali-tangga | 64 / 120 / 288 / 81 | 64 / 120 / 288 / 81 |
| Residual median / P95, piksel native | 0,6748 / 2,4734 | 0,6828 / 2,4951 |
| Lonjakan terbesar, kecepatan relatif / median lintasan | 407,72 | 1,44 |
| Shared track pada empat transisi bermasalah | 0 / 0 / 12 / 12 | 822 / 137 / 809 / 779 |

Lonjakan pada transisi yang sama turun menjadi sekitar 0,95–1,05 kali median. Semua transisi kandidat diperiksa untuk mencari lonjakan baru; hasil maksimum 1,44 kali median. Angka ini diagnostik konsistensi, bukan kecepatan fisik atau akurasi meter. Dua frame yang masih tidak terdaftar berada pada 124,506667 dan 124,956667 detik. Seluruh frame pantry, koridor dan workspace IT masuk world yang sama.

Probe awal gagal pada akses field resource dalam skrip diagnostik dan diperbaiki sebelum uji lengkap. Verifikasi ulang pertama selesai, tetapi guard inventaris menghentikan pemetaan karena salinan file utama SQLite tidak mencakup record WAL terakhir. Snapshot dipulihkan dengan SQLite backup; seluruh 153.735 record, byte fitur/keypoint/descriptor dan raw matching divalidasi identik dengan baseline, lalu semua hasil verifikasi dipakai ulang tanpa komputasi ulang. Attempt gagal tetap tersimpan; data sumber dan produk lama tidak diubah.

Kandidat `lantai2_part4_refine_4e1474fc_S03M_snapshot_9a363bea_A001` telah diperiksa dan diekspor sebagai kamera pinhole melalui `lantai2_part4_refine_4e1474fc_S03E_ready_cd68879d_A001`. Kontrak input inference memvalidasi pose, kalibrasi, identitas frame, dimensi dan hash seluruh 553 gambar undistorted. Bukti privat: `outputs/maintenance/lantai2-part4-camera-20261026/qa-calibrated/` (lokal).

Job E2E lanjutan `lantai2_part4_calib_1026` sudah terdaftar dan memakai audit/kamera hasil koreksi milik video yang sama. Pada checkpoint ini worker/controller melaporkan `waiting_resources`: GPU resmi index 1 dipakai Strata dan tidak memenuhi minimum 14 GiB bebas. Ini snapshot resource, bukan status live server. Setelah GPU tersebut tersedia, lanjut dengan `rtk proxy .venv/bin/python scripts/run_pipeline.py resume --job-id lantai2_part4_calib_1026`.

Belum ada depth/mesh/denah/HTML baru untuk take ini. Review kualitas tetap pending/unassessed dan skala relatif. Eksperimen ini belum menjadi fitur re-verifikasi kalibrasi otomatis pada pipeline default; pengujian geometri dari kamera baru dan validasi lintas rekaman masih diperlukan.

## Global architecture — 26 Oktober 2026

Branch `feature/global-architecture-alignment` menambahkan S06G setelah render komponen. Shared-frame camera centers dipasang dengan Umeyama Sim(3), lalu diperiksa dengan frame validasi terpisah, orientasi, conditioning dan cycle. GLB memakai `root_viewer × component_to_root × inverse(component_viewer)`; fixture memeriksa koordinat dan warna pada GLB asli. Kandidat gabungan hanya memuat world yang terhubung ke root; tahap ini belum global TSDF atau hole filling.

Kandidat terbaru `lantai3_aligned_1026/v001` selesai dari rekaman yang sama: 824/999 keyframe processed, 175 gap, satu grup gabungan (`component_001` + `component_003`) dan 10 world fallback. World utama belum tersambung; S06G memilih grup lain yang lolos validasi, tanpa memaksa posisi world utama. Dua area memakai 20 shared cameras: 14 fitting dan 6 validasi. Validation center max 0,0547% bentang lintasan bersama dan orientation max 1,615°. Ini konsistensi kamera, bukan akurasi meter/surface ground truth.

Delivery membawa `mesh/architecture.glb`, denah gabungan dan HTML offline dengan 11 pilihan. Semua 824 frame-ID tetap terwakili, dan 12 GLB lokal identik byte dengan kandidat dense sebelumnya. Sebanyak 41 artefak produk diverifikasi SHA/size; browser QA offline/mobile lulus tanpa error. Review tetap pending/unassessed; inspeksi manual source/render world utama masih menemukan bidang putih/floaters dan lubang, termasuk sekitar 215–224 detik. **Full-building reconstruction belum tercapai.**

Benchmark kamera CPU-only sudah selesai: baseline 370/502 keyframe, 18 accepted maps, largest map 79/502 (15,7%); dense 824/999, 16 accepted maps, largest map 287/999 (28,7%). Pipeline memilih 12 world lokal yang menambah coverage, lalu dua digabung sehingga HTML memiliki satu grup + 10 fallback. Reuse seluruh depth/mesh untuk tahap grouping terakhir; tidak ada inference ulang pada `lantai3_aligned_1026`. Profil opt-in yang sama disimpan pada `configs/pipeline.dense.yaml`.

Regresi terakhir connected-group: 282 tes Python, 279 lulus/3 skip; 12 tes alignment/GLB juga lulus di runtime geometri server yang dipin. Tujuh tes JS kalibrasi sebelumnya tetap berlaku untuk kode JS yang tidak berubah. Jalur positif GLB → denah → HTML juga diperiksa dengan fixture sintetis dan QA browser offline/mobile.

Uji reuse nyata `lantai3_global_1026` selesai tanpa inference baru: 370/502 frame, 17 komponen, tanpa kegagalan komponen, browser QA offline/mobile lulus. Hasil alignment `no_alignment`: tidak ada shared frame-ID pada 17 world yang masuk delivery. Tidak ada `architecture.glb` dummy. Laporan baru: `end_product/lantai3_global_1026/v001/reports/global-alignment.json` (privat/lokal). Hasil ini belum perbaikan visual atau full-building reconstruction; review pending, skala relatif.

Reuse sempat menolak world utama karena konfigurasi anotasi ditimpa worker duplikat pada sesi sebelumnya. Byte asli berhasil direkonstruksi dan dicocokkan tepat dengan SHA pada manifest surface sebelum dipulihkan. Byte yang berubah dibackup; mesh dan delivery final lama tidak diubah.

Regresi integrasi awal: 280 tes Python dijalankan, 277 lulus dan 3 skip backend; termasuk tes GLB nyata menggunakan pin Trimesh 5.1.1 di direktori dependency tes temporer, tanpa mengubah environment proyek. Tujuh tes JavaScript kalibrasi lulus. Uji kamera CPU `lantai3_dense_camera_1026` memakai frame asli 4 fps, sequential overlap64, SIFT longest-edge1600/max8192features; perbandingan dan kandidat hasilnya dicatat di atas.

## Riwayat sebelum 26 Oktober 2026

**Dokumentasi diperbarui 16 Oktober 2026.** Stack aktif: [TECHSTACK](TECHSTACK.md); prioritas: [ROADMAP](ROADMAP.md). Pengguna melaporkan job `lantai3_part1_1016_1525` selesai, overall OK. Status tetap completed/review pending/unassessed; pemisahan arsip tidak menjadi keputusan kualitas visual.

Angka resource, test count dan status run di bagian riwayat adalah snapshot pada tanggal eksperimen, bukan keadaan live server. Bukti regresi dashboard terakhir: verification.json (`../outputs/dashboard_qa/verification.json`, lokal),246tes lokal (242lulus,4dilewati),74tes server; input/progress/output desktop/mobile dan unduhan ZIP lulus. Hasil cache acuan tidak membuktikan keberhasilan rekaman independen.

## Room tour ArchiCon —16 Oktober2026

Pengguna meminta desain HTML yang selaras dashboard dan merepresentasikan arsitektur/interior. Template kini memakai Carbon Gray100/Plex, sidebar komponen dengan thumbnail denah asli, viewport charcoal, kontrol potongan model, timeline kamera dan lembar denah terang. Kalibrasi/JSON berada pada tampilan denah; font dan lisensi OFL embedded. Header menghitung252frame-ID unik, sedangkan hitungan kamera per world tetap ditampilkan sesuai data (dapat beririsan).

Hasil `end_product/lantai3_part1_1016_1525/v002` dibangun dari bundle terverifikasi tanpa inference baru. Seluruh embedded data mesh/kamera/denah identik dengan v001;16artefak selain HTML identik. Sebanyak225file produk sebelumnya tetap cocok SHA; hanya latest pointer job berubah menjadi v002. Packet review baru mengikat `review-bundle-v002.json`; review/bundle v001, stage numerik, registry dan durasi rekonstruksi dipertahankan. Status tetap completed/review pending/unassessed.

QA offline seluruh4komponen, playback/gap stop/lookaround, keyboard tab, evidence overlay, pengukuran/kalibrasi/JSON/SVG serta lebar320–1440px lulus tanpa pageerror/overflow. HTTP/CSP memuat font embedded dan ZIP terbaru cocok v002. Suite lokal258tes:254lulus/4dilewati;7tes JS kalibrasi lulus. Fixture audit historis pada satu tes kini dibaca dari arsip jika path aktif tidak tersedia, sehingga pemisahan arsip tidak membuat regresi tersebut terlewati. [Panduan dan bukti](ROOM_TOUR.md).

## Pemisahan eksperimen closed —16 Oktober2026

Pengguna menyetujui arsip dingin setelah analisis dependensi. Sebanyak83entri/2.161GiB/2.988file di laptop dan141entri/12.768GiB/15.728file di server dipindah dengan rename no-overwrite ke `outputs/archive/experiments_2026-10-16/{flat,jobs}/`. Semua file arsip cocok SHA/mode/mtime/struktur dengan manifest sebelum move; tidak ada data eksperimen dihapus. Manifest asli dan registry historis tidak diedit; resume/reuse/refinement lama perlu restore. Produk final tetap di end_product, termasuk PART1. Delapan versi final yang belum ada di server dibackup dan dicek terhadap manifest; kini18versi cocok pada kedua host. [Panduan restore](OUTPUTS_ARCHIVE.md).

Job pengguna tetap completed/review pending/unassessed. Seluruh produk, numerical artifacts, konfigurasi/review/status job dan registry tetap cocok hash; satu `dashboard-snapshot.json` berubah karena monitoring saat lock arsip dipegang, bukan perubahan rekonstruksi. Diagnosis SSH dari satu status probe yang terhalang sandbox dipertahankan di evidence audit; retry dengan akses jaringan berhasil. History ArchiCon hanya memuat job pengguna. Cek90artefak bundle,15registry+target stage server,17artefak final via HTTP, ZIP CRC dan rebuild HTML dengan SHA identik lulus tanpa inference atau layanan agent.

Regresi baru:257tes lokal,253lulus/4dilewati;11tes arsip server lulus, termasuk apply/restore, interruption recovery, no-overwrite, path/symlink protection, hash mismatch dan lock aktif. Bukti: verifikasi pelaksanaan (`../outputs/documentation_audit/outputs-retention-2026-10-16/execution-verification.json`, lokal). Review visual formal dan penyelarasan desain output HTML belum dilakukan dalam pengarsipan.

## Dashboard ArchiCon

Dashboard localhost dark mode mengikuti DESIGN.md dengan font IBM Plex lokal:
input video/upload streaming, history legacy/nested, progress stage nyata dan
preview QA, hasil room tour/denah/mesh serta paket ZIP. Controller detached memakai
kernel lock yang diwariskan sehingga restart browser/backend tidak membuka job kedua.
Review tetap manual, tidak ada layanan model/agent otomatis. Command dan batas
operasional: [ARCHICON.md](ARCHICON.md).

Uji HTTP nyata `archicon_cache_001` memakai cache terpercaya dari video asli,
bukan inference fresh: hasil `end_product/archicon_cache_001/v001`,286/302frame,
dua komponen, browser QA lulus, review pending/unassessed. Desktop/mobile dashboard
lulus tanpa pageerror/overflow; bukti di `outputs/dashboard_qa/`. Rekaman independen
kemudian selesai menurut pengguna, overall OK; verifikasi pengarsipan memeriksa integritas dan akses produk, bukan review geometri.


## Perapian folder job —16 Oktober2026

Job baru memakai layout_version2: `outputs/jobs/<job>/config`, `stages/<stage>/<run_id>`,
`logs`, `qa` dan `review`. Produk di `end_product/<job>/vNNN` dengan latest pointer.
Run registry immutable `outputs/.run-index` ditransfer berdasarkan hash; numerical
stage CLI memakai resolver yang juga mendukung direktori datar lama.
Dokumen utama pindah ke docs dengan pointer root; README utama diringkas, catatan
pengembangan historis disimpan pada DEVELOPMENT.md. Tidak memindahkan/menghapus
hasil legacy atau mengubah PART1.

Regresi nyata: `layout_cache_001` berhasil packaging/QA286frame tanpa inference;
aplikasi JSON test needs_human_review membuat v002 dengan byte produk identik,
bukan penerimaan kualitas visual. `layout_smoke_001` menjalankan seluruh stage
fresh60/60 dengan registry, konfigurasi/log/stage nested, QA offline/mobile lulus.
Resume job nested maupun legacy diperiksa. Smoke baru tetap review pending.
Bukti perapian akhir: `outputs/jobs/layout_smoke_001/layout-verification.json`.

## Uji numerik fresh dan refinement nyata

Pengguna menghentikan Strata sementara; RAM available naik menjadi48,96GiB.
Pipeline tetap memakai hanyaGPU0 dan batas resource yang sama.

- `part2_smoke_001`: resource gate lolos, tetapi klip25–37s didominasi dinding
  lorong dan SfM menolak seluruh initial pair; tidak ada peta palsu yang diteruskan.
- `part2_smoke_002`: registrasi dilakukan sebelum transcode selesai; checksum
  menolak file yang berubah. Run gagal dipertahankan sebagai diagnosis.
- `part2_smoke_003`: klip15–45s, durasi30,013317s,60/60keyframe. Keenam stage
  audit/kamera/DA3/TSDF/render/export benar-benar dijalankan tanpa reuse rekonstruksi.
  HTML offline/mobile lolos QA; peak sampledRSS2,422GiB danGPU6,217GiB.
- Review manual meminta dua kandidatS06 dengan kamera/depth tetap. Kedua kandidat
  menjalankan surface/render/export dan QA nyata; tidak ada kamera/DA3/GPU rerun.
  Stricter menambah hole lokal, coarser tidak menunjukkan peningkatan material.
  Baseline dipertahankan pada `end_product/part2/part2_smoke_003/v003`, kualitas
  `accepted_partial` hanya untuk scene klip/draft relatif, bukan seluruh ruangan.
- `part2_full_001`: video penuh melalui18stage numerical tanpa reuse rekonstruksi,
  302keyframe/280processed/22gap, lima world terpisah, QA offline/mobile/semua
  komponen lulus. Ini bukti cold-run video→mesh/denah/HTML, bukan hasil terbaik.
- Matching sequential membuat peta lebih terpecah daripada baseline exhaustive
  lama dengan feature budget sama. Default kini `matching:auto`: exhaustive
  hingga400keyframe aktual (maks79.800pasangan), sequential di atasnya. Explicit
  override tersedia. Suite lokal228tes,224lulus/4skipped setelah perubahan.
- `part2_full_002`: perbandingan matcher selesai; hanya audit fresh run001 dipakai
  ulang, kamera/depth/mesh baru.283/302processed,19gap, satu world utama283frame
  dibanding world terbesar sequential158. Export mesh detail4.090.024triangle dan
  viewer582.237triangle. HTML offline/mobile lulus, review manual `needs_capture`;
  hasil `end_product/part2/part2_full_002/v002`. Tidak mengklaim semua interior lengkap
  atau lebih lengkap dari arsip286/302yang masih memiliki entrance-local.
- Ditemukan metadata kandidat yang kehilangan diagnosis peta kamera ditolak saat
  reuse. Perbaikan meneruskan `failed_components` upstream ke job/bundle kandidat.
  Tes mereproduksi kegagalan sebelum fix; probe nyata `part2_metadata_001`
  memeriksa preservasi diagnosis dengan hanya exporter/QA, tanpa inference baru.
  Suite final229tes,225lulus/4skipped; hasil server dan checksum dicatat pada
  `outputs/part2_full_002/final-verification.json`.

Bukti: `outputs/part2_smoke_003/verification.json`, `REFINEMENT_REVIEW.md`,
`refinement-comparison.json`, `server-execution.json`; full cold-run:
`outputs/part2_full_001/verification.json`. Timestamp smoke adalah waktu klip;
`input-derivation.json` menyimpan offset15s terhadap video acuan.
Pada tahap eksperimen di atas belum ada rekaman independen atau ukuran lapangan; laporan uji independen berikutnya dicatat di bagian terbaru. Tidak ada klaim semua interior lengkap.

## Riwayat — 15 Oktober 2026

## PART2: orchestrator dan review manual

Update setelah pengguna meminta lanjut: review manual visual baseline selesai.
Laporan lengkap (`../outputs/archive/experiments_2026-10-16/flat/part2_cache_001/MANUAL_REVIEW.md`, lokal) mencatat tujuh temuan:
entrance-local fragmentaris, interior berkaca belum terbukti lengkap, objek area
kerja melebur/tertutup orang, gap akhir, cacat lokal lorong, denah belum semantik,
dan skala belum diukur. Keputusan `needs_capture`; versi v003 menambahkan review
pada byte produk v002 yang sama. Tidak ada klaim perbaikan geometri/refinement.

Resource server diperiksa lagi: RAM available sekitar 6 GiB, gate 8 GiB belum
terpenuhi. Pemakai terbesar `strata` PID3052441 di `/srv/other-project`,
RSS/PSS sekitar41,78GiB; parentPython3052421, launcherstart-strata.sh3052419.
Informasi disampaikan kepada pengguna yang akan mengosongkan kapasitas.
Tidak menghentikan proses lain, menurunkan gate atau memakai GPU1.

`scripts/run_pipeline.py` menyediakan run/plan/status/resume/prepare-review/apply-review.
Worker server detached menyimpan state dan attempt; source/input/output diverifikasi
hash, hanya GPU0 terpilih, CPU stage menyembunyikan CUDA. Pose ekstrem dikarantina
per frame; data failure komponen dilaporkan tanpa membuang komponen valid lain.

Keputusan pengguna: **review manual, jangan otomatis karena usage limit**.
Seluruh fungsi pemanggilan Codex reviewer otomatis dihapus. Pipeline hanya
menyiapkan gambar, prompt, schema dan fingerprint. Review JSON dapat menerapkan
preset S06/S07 maksimal 2 ronde × 2 kandidat, lalu berhenti lagi untuk review manual.
Resume transaksi refinement/resource wait diuji tanpa pemanggilan model.

Regresi cache nyata `part2_cache_001` memakai video dan kamera/depth/mesh lama
yang lolos validasi. Delivery v002 (`../end_product/part2/part2_cache_001/v002/README.md`, lokal)
berisi GLB detail, denah dan HTML offline; `review_status=pending`,
`quality_status=unassessed`, `scale_status=relative`. 302 keyframe, 286 processed,
16 gap; bukan rekonstruksi seluruh interior lengkap. v001 adalah snapshot sebelum
kebijakan manual baru dan tetap disimpan; PART1 tidak diubah.

Browser QA v002: offline reload, dua komponen, denah, mobile 390px tanpa overflow
dan tanpa page error. Simpan→reset→muat kalibrasi JSON diuji nyata di browser;
kalibrasi satu komponen tidak mengubah komponen lain. Jarak uji sintetis hanya
untuk QA, bukan ukuran lapangan dan tidak diterapkan pada default delivery.

Smoke fresh `part2_smoke_001` memakai klip 12 detik turunan video acuan.
Audit selesai; tahap kamera tertahan `Host MemAvailable below required 8 GiB`.
Resume menghasilkan `waiting_resources`, tidak menurunkan gate atau memakai GPU1.
Belum ada bukti numerik fresh sampai output atau video independen. Kegagalan semua
komponen belum menghasilkan paket visual lengkap; periksa log diagnosis.

Hasil tes dan checksum regresi sebelum review disimpan di
`outputs/part2_cache_001/verification.json` setelah verifikasi akhir.
Checksum v003 dan penerapan review: `outputs/part2_cache_001/manual-review-verification.json`.
Uji manual refinement numerik nyata, PnP generik, semantic floorplan, anotasi kaca
dan dimensi terverifikasi masih pekerjaan berikutnya. Lihat [WORKFLOW](../WORKFLOW.md).

## Riwayat — 14 Oktober 2026

## Hasil historis — 14 Oktober: seluruh timeline video acuan diproses

Seluruh302keyframe dicoba untuk kamera.283frame pada global map utama dan3frame
entrance-local sudah melalui depth, filtering, TSDF dan actual-triangle rendering.
Enam belas frame tanpa pose valid ditampilkan eksplisit pada video review, bukan
dihapus. Semua delapan area timeline mempunyai view terproses; interior unseen
dan sambungan area masuk tetap belum lengkap.

GLB environment utama — 76,76MiB (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/room-full.glb`, lokal) ·
GLB area masuk — world lokal terpisah (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/entrance-local.glb`, lokal) ·
Video review seluruh take — 7,29MiB (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/full-video-review.mp4`, lokal) ·
Laporan cakupan (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/report.json`, lokal).

Bukti visual Overview full-video mesh (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/overview.png`, lokal)

| Area timeline | Keyframe dicoba | View dirender |
|---|---:|---:|
| 0–7s, area masuk | 14 | 4 |
| 7–24s, akses kantor/open area | 34 | 34 |
| 24–56s, lorong | 64 | 63 |
| 56–82s, area belakang/jendela | 52 | 52 |
| 82–117s, meja kerja berpenghuni | 70 | 70 |
| 117–120s, penyimpanan/transisi | 6 | 6 |
| 120–143s, koridor/ruang berkaca | 46 | 46 |
| 143–150,84s, foyer/tangga | 16 | 11 |

Jumlah view bukan persentase ruangan lengkap. Ruang berkaca hanya terlihat dari
koridor; bagian di balik orang/furnitur dan tangga lebih jauh tidak teramati.
Prioritas rekaman tambahan: **0–7s** sambil berjalan dan overlap ke koridor;
**120–143s** masuk ke setiap ruang berkaca; **148,79–150,84s** lanjutkan tangga
dengan overlap. Area berpenghuni82–117s sebaiknya direkam lebih kosong bila ingin
geometri meja/kursi yang tertutup orang. Frame55,93s gagal, tetapi tetangganya ada.

Koreksi penting: camera_full001 memiliki satu pose59,85s sejauh~75.000unit relatif,
meski video bergerak sedikit. Fixed-mapPnP memulihkannya dengan184inliers,
median1,347/P953,391px; langkah kamera tetangga0,124 dan0,085unit. Kamera/K/images
lain tetap sama; original map/database hashes identik. Derived camera component
baru dipakai untuk mengulang seluruh depth/fusion/render. Surface_full001 dan
render_full001 disimpan sebagai diagnosis, **tidak dipilih**.

Hasil terpilih: `camera_corrected_001` → `depth_full_002` → `surface_full_002` →
`render_full_002`, ditambah entrance-local. Mesh utama3.964.088triangle/2.057.281
vertices; strict comparison3.530.113triangle. Bounds kembali pada extent ruangan
yang masuk akal, bukan outlier puluhan ribu unit. Semua komponen mesh tetap ada;
tidak ada smoothing/fill/prune. Coverage mode mempertahankan depth unsupported
yang tidak contradicted sebagai uncertain. Warna dan detail masih kasar.

GPU inference main peak sampled6,217GiB; host2,60GiB;24windows16view. Corrected
fusion CPU peak4,078GiB; render1,114GiB, tanpaGPU. Seluruh pekerjaan memakaiUUIDGPU0
untuk DA3 dan4CPUthreads; GPU1 tidak dipakai. Higher-resolution sequentialSfM dicoba
tetapi memecah peta dan tidak memulihkan gap; tidak dipilih.

166tests server geometry lulus; laptop163lulus/3backendskipped. Video final memiliki
302frame,560×566,2fps/151detik; source timestamp asli ditampilkan. Presentasi ini
tidak mengubah durasi150,84s/timingPTS sumber. Status paket `partial_reconstruction_delivered`:
all-frame accounting selesai, **full-room reconstruction complete=false**.

Bukti visual Source/render sepanjang peta utama (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_full_delivery_001/full_path_review.jpg`, lokal)

## Riwayat: audit lubang dan fusion pilot

Audit triangle asli dan lima variasi cached-depth selesai. Kandidat sementara
**voxel1,5× / truncation8vox / confidenceq10** mengurangi lubang kecil pada dinding.
Depth,52 TRAIN, kamera global dan semantic masks tetap; DA3 tidak dijalankan ulang.

GLB utama (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_fusion_review_001/room-pilot-improved.glb`, lokal) (27,16 MiB) ·
Alternatif coverage q0 (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_fusion_review_001/room-pilot-coverage-q0.glb`, lokal) (19,24 MiB) ·
Review lengkap (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_fusion_review_001/review.md`, lokal) ·
Angka dan input hashes (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_fusion_review_001/comparison.json`, lokal).

Bukti visual Original, reference, dan lima variasi (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_fusion_review_001/comparison.png`, lokal)

| Ukuran pada delapan view development | Reference | Kandidat 1,5×/8vox/q10 |
|---|---:|---:|
| Mean triangle-ray coverage | 89,85% | 91,16% |
| Visibility crop dinding kiri | 98,98% | 100,00% |
| RGB MAE common-hit /255 | 15,787 | 15,975 |
| Triangle count | 3.767.043 | 1.407.316 |
| Components | 12.519 | 1.524 |
| Boundary length unit relatif | 3.078,95 | 790,11 |

Crop100% bukan seluruh dinding selesai. Boundary length turun74,34%, tetapi
mencakup perimeter/intentional exclusions dan berubah saat tessellation lebih
kasar; bukan persentase lubang tertutup. MAE sedikit memburuk. Area kosong besar,
blur warna dan risiko bridging dari truncation lebar tetap ada. Belum ground truth
atau skala meter.

q0 mendapat coverage96,25%; dibanding q10 pada voxel2×/trunc4 identik, components
naik2,73× dan MAE naik0,537/255. Simpan sebagai alternatif coverage, belum geometri
utama. Delapan query tetap tidak dimasukkan ke depth/map/fusion, tetapi kini dipakai
memilih hyperparameter; evaluasi akhir butuh unseen views.

Double-sided menambah mean internal coverage hanya0,052percentage point; speckle
internal adalah missing geometry. Dari luar, double-sided membantu menampilkan
permukaan belakang. Review GLB memakai satu rigid root transform dan double-sided
material; binary geometry tetap identik dengan source. Semua run lama dipertahankan.

Kelima fusion dan keenam audit final exit0; CPU4thread, sampled fusion peak≤1,82GiB,
GPU process usage0. Runtime/watchdog/source snapshots tercatat per run.
Reference001 mendahului penguatan provenance; reference002 dipakai comparison.
Dukungan K integer ditambahkan setelah audit, tanpa mengubah input K float aktual.

Berikutnya: bounded higher-resolution depth benchmark pada subset TRAIN, confidence
adaptif/coverage dan high-resolution RGB. Full take/optimized splat masih pending.

## Riwayat: depth pilot dan TSDF awal

Seluruh 52 frame training pilot 15–45 detik sudah diproses dalam empat window 16, stride 12.
64 prediksi mentah dipertahankan;52 output unik dipilih dari window terdekat pusatnya,
tanpa merata-ratakan depth atau menggabungkan translasi chunk. Global camera tetap.

| Run | Hasil |
|---|---|
| `l2bp3_depth_resource_001` | Satu window 16 lolos sebelum full pilot |
| `l2bp3_depth_pilot_001` | Empat window,52 depth unik;GPU process sample 6,217GiB;host peak 3,857GiB |
| `l2bp3_masks_002` | Polygon glazing yang ditinjau + 52 person boxes manual di 33 frame + 1 detector box;mask tetap incomplete |
| `l2bp3_filter_002` | Mean coverage 87,09% setelah spatial masks → 85,38% setelah multi-view; tanpa flag kehilangan >50% base |
| `l2bp3_tsdf_001` | Dua mesh triangle berwarna PLY/GLB,CPU 51,74detik,peak RAM 3,318GiB |

GLB untuk review (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_tsdf_001/room-pilot.glb`, lokal) (96,7 MiB) ·
GLB world asli (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_tsdf_001/filtered.glb`, lokal) (73,7 MiB) ·
PLY (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_tsdf_001/filtered.ply`, lokal) (144,3MiB) ·
Report (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_tsdf_001/report.json`, lokal)

Bukti visual Perbandingan TSDF (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_tsdf_001/reports/mesh_review.png`, lokal)

| Surface | Vertices | Triangles | Components | Bagian triangle pada komponen terbesar |
|---|---:|---:|---:|---:|
| Spatial baseline | 2.130.408 | 3.997.957 | 12.969 | 91,74% |
| Multi-view filtered | 2.005.959 | 3.767.043 | 12.519 | 92,81% |

Struktur lorong, lantai, dinding dan plafon terlihat, tetapi mesh masih kasar dan
terbuka: filtered memiliki 250.095boundary-or-nonmanifold edges dan vertex-manifold
false. Semua komponen dipertahankan. Ini bukan mesh collision/watertight atau hasil
optimized splat. Warna vertex berasal dari 280×504processed RGB, belum texture atlas
resolusi tinggi. Jumlah triangle bukan bukti akurasi/detail. Preview menampilkan
100k sampled surface vertices, dengan transform display global tercatat; geometri
asli tidak diubah. GLB review memakai satu transform rigid global menuju estimated
Y-up; transform dicatat pada `reports/viewer_export.json`, tanpa konversi scale
atau koreksi per-frame. Scale tetap relatif.

Overlap 12 frame memiliki median depth ratio 0,975–0,995 dan 98,5–99,8%pixel agreement
dalam toleransi 10%. Shared four-view smoke dibandingkan pada frame/RGB/K yang sama.
Root memverifikasi seluruh 64 raw depth/scale sama persis dengan output aligned dan
w2c global. Ini konsistensi internal model, bukan ground truth.

Filtering memakai validity, conservative people/glass exclusions, confidence q10,
relative 8%depth-edge threshold, lalu empat temporal neighbors dengan toleransi Z 12%.
Permukaan tertutup objek dicatat occluded; tidak dihitung sebagai contradiction.
Unsupported coverage juga dicatat terpisah. Normals unit menghadap kamera tersimpan.
Detektor SSDLite hanya menemukan satu orang; 52 bbox manual ditambahkan setelah
inspeksi agar orang kecil/terhalang tidak dianggap terdeteksi otomatis.

Open3D 0.20 gagal pada fixture plane karena regresi block activation float32 depth.
Environment `.venv-geometry` terpisah memakai 0.19.0; fixture plane serta semua 119 tes
server lulus. Laptop 119 tes selesai,dua backend tests dilewati. Inference tetap pada
runtime asalnya. [Laporan upstream](https://github.com/isl-org/Open3D/issues/7592)

Review mesh (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_tsdf_001/reports/manual_review.md`, lokal) ·
Coverage filter (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_filter_002/report.json`, lokal) ·
Depth windows (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_depth_pilot_001/report.json`, lokal)


Riwayat tahap sebelumnya di bawah tetap dipertahankan. Kamera/fusion memakai CPU;
DA3 memakaiGPU0. GPU1,
source proyek lama, serta checkpoint cache tidak diubah.

| Tahap / run | Hasil | Batas klaim |
|---|---|---|
| `preflight_004` | Resource pass; empat snapshot cache ada dan ukuran/config sesuai; PyCOLMAP 4.2.1 tercatat | Snapshot sebelum smoke: `inference_ready=false`; kernel splatting belum diuji |
| `l2bp3_audit_local_002` | 9.045 source frame → 302 keyframe; 242 train/60 validation; portrait 1080 × 1920 | Sharpness/exposure merupakan proxy; semantic masks belum dibuat |
| `legacy_l2bp3_001` | 429 cache frame dipetakan ke JPEG; sembilan spot check korelasi >0,999 | Timestamp filename perkiraan; raw scale relatif; visual-only |
| `l2bp3_camera_pilot_002` | 52/52 train frame diregistrasi; median 0,721 px, P95 2,487 px native; 3.658 sparse points | Scale relatif; delapan validation view dilokalisasi pada run terpisah; belum mesh |
| `l2bp3_validation_opencv_002` | 8/8 fixed-map query pose; median0,965px/P953,270px; map/database utuh | Localization tidak membuktikan physical calibration atau kualitas rendering |
| `l2bp3_camera_comparison_002` | SIMPLE_RADIAL juga lolos; OPENCV dipertahankan sementara | Track/inlier berbeda; cross-model trajectory masih berbeda |
| `l2bp3_da3_smoke_001` | DA3-LARGE lokal,4×504×280;depth seluruhnya finite/positif | Empat-view runtime/kontrak numerik, bukan depth pilot penuh atau mesh |

## Input dan seleksi

Source `raw_videos/lantai2_B_part3.MOV`, durasi 150,84 detik, hash
`2247b1b154b1f26dd4f85e90ca3a5d8afb28a73bf76348964448c1b44cff1955`.
Salinan server identik. Rotasi display diterapkan sekali dan frame memakai decoder
PTS aktual. Sampling memilih citra paling tajam per window 0,5 detik. Sebanyak 34
frame berskor sharpness relatif rendah dipertahankan; gap maksimum 0,968334 detik.

Split tetap berdasarkan blok temporal 2 detik, seed 7. Pilot 15–45 detik berisi
52 train/8 validation. Lima halaman contact sheet pilot telah ditinjau; segmen
belokan dan dinding rendah tekstur dipertahankan. Kaca/orang dicatat untuk masks.
Stress segment 110–140 detik punya contact sheets, namun belum direkonstruksi.

- Audit video (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_audit_local_002/reports/video_audit.md`, lokal)
- Review capture (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_audit_local_002/reports/manual_review.md`, lokal)
- Baseline legacy dan keterbatasan (`../outputs/archive/experiments_2026-10-16/flat/legacy_l2bp3_001/baseline_review.md`, lokal)

## Kamera pilot

PyCOLMAP CPU 4.2.1, OPENCV single-camera, native RGB, feature extraction longest
edge 960, exhaustive matching terbatas 52 train frame, lalu bundle adjustment.
Ambang residual dinyatakan dalam **native image pixels**, bukan feature pixels.
Run terakhir punya 22.501 observations; median 0,721252 px, P95 2,487344 px.
Dua kandidat model saling overlap; candidate count tidak berarti jumlah komponen
graf terputus. Connectivity graf belum dihitung terpisah.

Focal hasil BA sekitar fx=890,58/fy=889,36 pada gambar 1080 × 1920. Lens 0,5x tetap
berdasarkan keterangan pengguna; focal hasil fitting bukan verifikasi metadata
lensa atau akurasi metric. RGB undistorted berukuran 1078 × 1927 dengan K baru,
cx=539/cy=963,5. Pose diekspor OpenCV c2w, +x kanan/+y bawah/+z depan, world relatif
tanpa gravity/yaw alignment. Native K tidak boleh dipakai menggantikan K undistorted.

Plot menunjukkan belokan lalu lintasan memanjang, sesuai urutan capture. Tidak
terlihat loncatan terisolasi pada speed relatif; gap train akibat blok validation
menjelaskan dua langkah panjang. Ini review kontinuitas, bukan pengukuran drift
terhadap ground truth. Cloud dalam plot dibatasi percentile 1–99%; file sparse
asli tetap lengkap.

Bukti visual Lintasan kamera dan sparse map (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_camera_pilot_002/reports/camera_review.png`, lokal)

- Report pilot terakhir (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_camera_pilot_002/report.json`, lokal)
- Kamera undistorted (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_camera_pilot_002/undistorted/cameras.json`, lokal)
- Plot SVG (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_camera_pilot_002/reports/camera_review.svg`, lokal)
- Preflight terbaru (`../outputs/archive/experiments_2026-10-16/flat/preflight_003/preflight.json`, lokal)

Pilot terakhir selesai sekitar 12,23 detik, child peak RSS 724.640 KiB (~0,69 GiB),
empat thread dan `CUDA_VISIBLE_DEVICES=""`. Angka ini khusus pilot CPU, bukan
estimasi kebutuhan DA3/full take. Run 001 dan 002 sama-sama melewati gate; sparse
point count/K sedikit berbeda pada mapping multithread walau seed disimpan.

## Validation dan kalibrasi alternatif

OPENCV dan SIMPLE_RADIAL sama-sama meregistrasi52/52train frame dan menerima8/8
validationquery pada map tetap. OPENCV terbaru:1.530inliers,median0,965/P953,270px;
radial:1.537inliers,0,964/3,315px. Tidak ada BA, query triangulation, atau optimasi
scene memakai validation RGB. Original database/map hashes tetap identik.
OPENCV validation di-undistort pada K/grid training yang sama; empat OpenCV thread
sudah ditegakkan dan diverifikasi. Native-only radial juga tersedia.

Model berbeda memakai track/inlier berbeda. Alignment lintasan berbasis orientation
masih berbeda rotation median2,46°/P957,45°, sehingga physical calibration/drift
belum tervalidasi. OPENCV menjadi pilihan sementara karena train residual lebih
rendah, heldout residual hampir sama, serta rectified grid siap.

Perbandingan kamera (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_camera_comparison_002/comparison.md`, lokal) ·
Validation OPENCV (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_validation_opencv_002/report.json`, lokal) ·
Validation radial (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_validation_radial_001/report.json`, lokal)

## Smoke DA3 yang benar-benar dijalankan

DA3-LARGE snapshot lokal dimuat pada `.venv-inference`, source vendor part2,
NumPy1.26.4/PyTorch2.11.0+cu130. Environment merupakan salinan runtime lama dan
masih mewarisi package sistem/user; bukan environment hermetik. Bobot tidak diunduh
ulang. API/source/weightsSHA256 tercatat pada manifest. GPU0 UUID diisolasi menjadi
logicalcuda:0; GPU1 tidak digunakan.

Input empat train view20,08/22,15/24,02/25,58detik memakai RGB undistorted dan w2c/K.
Output empat depth/confidence280×504 serta processedRGB tersimpan pada NPZ.
Seluruh depth finite/positif. API alignment scale0,2773489702; depth dibagi scale
tersebut agar mengikuti unit map input. Raw predicted E/K disimpan sebelum API
menggantinya dengan input conditioning. Kesamaan output E dengan input bukan
bukti akurasi AI; depth tetap optical-axisZ relatif, bukan meter.

Peak PyTorch allocated2,665GiB/reserved3,561GiB; host ru_maxrss3,854GiB;
supervisor sampledGPUprocess3,875GiB. Empat CPU thread, satu OpenCV thread, dua
preprocessing thread. Model forward~0,97detik; seluruh supervisor run~12,89detik.
Process keluar0 tanpa abort dan GPU0 tidak meninggalkan compute process. Peak
sampled bukan batas atomik atau jaminan window 16/fulltake muat.

Bukti visual Smoke RGB, depth, confidence (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_da3_smoke_001/reports/depth_review.png`, lokal)

Dinding/lantai/plafon dan furniture besar terlihat pada depth. Kaca, orang, objek
tipis, dan batas diskontinuitas belum diberi masks. Confidence mentah1–9,7 bukan
probabilitas terkalibrasi. Median relative Z disagreement terhadap sparse training
tracks3,0–10,7% perview merupakan diagnostik internal, bukan ground-truth error.

Report smoke (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_da3_smoke_001/report.json`, lokal) ·
Review visual (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_da3_smoke_001/reports/manual_review.md`, lokal) ·
Diagnostik sparse Z (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_da3_smoke_001/reports/sparse_depth_agreement.json`, lokal) ·
Supervisor metrics (`../outputs/archive/experiments_2026-10-16/flat/l2bp3_da3_smoke_001/supervisor_metrics.json`, lokal)

## Verifikasi dan provenance

Suite sebelumnya166tes; tambahan denah/tour menambah6tes kontrak geometri dan
ekspor. Hasil fresh tersimpan di laporan verifikasi paket tour. Suite mencakup GPU remapping/resource gate,
PTS/rotasi/VFR, seleksi/split, cache legacy, pose/K/reprojection, run collision,
failure status, dan larangan memakai audit upstream yang belum selesai.
Tambahan audit triangle mencakup backside/farther front-face visibility,
GLB material/binary preservation, fixed-camera/map provenance dan ablation pair.
Review independen run 001 menghitung ulang residual dari binary COLMAP, memeriksa
invers pose, dan hash/dimensi seluruh native serta undistorted images.
Temuan status upstream dan pelabelan model kandidat sudah diperbaiki; run 002
membuktikan pipeline hasil perbaikan berjalan.

Run menyimpan source hashes, config, frame hashes, dan status. Repository lokal
serta server belum memiliki commit/remote. Output tidak masuk Git; source
allowlist disinkronkan terpisah dari artefak. Run audit 001 yang gagal akibat batas
parser FFmpeg disimpan dengan status failed; ekstraksi balanced expression sudah
diperbaiki dan diuji sebelum audit 002. Tidak ada overwrite run lama.

## Rencana pada 14 Oktober — rujuk ROADMAP untuk prioritas sekarang

1. Review denah draft dan HTML offline; ambil ukuran nyata segmen yang jelas,
   kalibrasi dan periksa beberapa jarak lain. Komponen masuk terpisah.
2. Review holes, double surfaces dan fragments pada mesh asli; evaluasi perubahan
   depth/masks/voxel dengan split dan viewpoint tetap. Belum prune/fill secara otomatis.
3. Benchmark depth resolution lebih tinggi secara bounded untuk detail, atau
   integrasikan high-resolution RGB pada optimized surface/splat setelah adapter siap.
4. Uji compiler/rasterizer Blackwell forward/backward sebelum training splat/mesh.
5. Orchestrator kini tersedia dengan review manual; uji fresh tertahan resource.
   Full take acuan diproses dengan gap yang dilaporkan; MapAnything masih opsional.

## Denah dan HTML offline —14Oktober2026

`outputs/l2bp3_room_tour_005` adalah paket terbaru. SVG/PNG menampilkan irisan
geometri aktual, kandidat garis berulang dan lintasan kamera. Input SHA diperiksa
terhadap render manifest; scene transform diterapkan sekali. Viewer utama585.455
segitiga,278.997vertex; sumber detail3.964.088segitiga tetap immutable.

Satu `room-tour.html` membundel Three.js, mesh dan bukti denah; tanpa backend/CDN.
283kamera utama +3pintu masuk, komponen tidak digabung. Kalibrasi browser adalah
draft per-komponen dan sesi, belum ada acuan lapangan. SVG unduhan menyimpan skala draft dan segmen pilihan.
Test kontrak mencakup orientasi, irisan, floor estimate, preservasi COLOR_0,
kalibrasi invalid/overflow dan observer di luar batas mesh. Run001 gagal akibat
konversi warna Trimesh;002–004 menyimpan iterasi sebelum perbaikan kanvas denah.
Tidak ada rerun DA3/GPU atau perubahan model/video/mesh sumber.
