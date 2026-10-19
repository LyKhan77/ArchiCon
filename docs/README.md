# Dokumentasi proyek

**Diperbarui:** 16 Oktober 2026. Dokumentasi aktif mengikuti pipeline E2E dan dashboard ArchiCon. Pengguna melaporkan rekaman baru selesai, overall OK; review formal tetap pending. Eksperimen lama dipisahkan sebagai arsip dingin tanpa penghapusan.

- [SETUP](SETUP.md): checkout source, contoh profil mesin dan batas data privat yang tidak masuk Git.
- [CHANGELOG](../CHANGELOG.md): checkpoint dan perubahan selesai; [AGENTS](../AGENTS.md): aturan repository.
- [TECHSTACK](TECHSTACK.md): teknologi aktual, runtime/versi, model, resource, format artefak dan batas implementasi.
- [PRD](PRD.md): kebutuhan dan batas klaim.
- [ROADMAP](ROADMAP.md): milestone dan pekerjaan tersisa.
- [PROGRESS](PROGRESS.md): riwayat eksperimen dan bukti eksekusi.
- [WORKFLOW](../WORKFLOW.md): command pengguna dan alur review manual.
- [ArchiCon](ARCHICON.md): dashboard web input, progress dan output.
- [Room tour](ROOM_TOUR.md): viewer arsitektur/interior dark mode, kontrol 3D/denah, kalibrasi dan HTML offline.
- Spec ArchiCon (`superpowers/specs/2026-10-16-archicon-dashboard-design.md`, lokal): desain yang sudah diimplementasikan.
- Rencana implementasi (`superpowers/plans/`, lokal): catatan bertanggal dan bukti closeout; bukan perintah menjalankan ulang pekerjaan.
- Arsip roadmap riset (`archive/ROADMAP-research-2026-10-14.md`, lokal): rencana awal P0–P6, termasuk kandidat yang belum diimplementasikan.
- [Arsip dan restore](OUTPUTS_ARCHIVE.md): lokasi eksperimen lama, manifest SHA dan pemulihan path untuk reuse/resume.
- [Analisis pemisahan output lama](OUTPUTS_CLEANUP_ANALYSIS.md): snapshot inventaris/dependensi laptop-server sebelum operasi.

Semua command dijalankan dari root proyek, kecuali disebutkan berbeda. File PRD/ROADMAP/PROGRESS pada root adalah pointer kompatibilitas. Source/doc mengikuti .sync-files; media/model/generated outputs ditransfer terpisah. Link artefak historis dapat memerlukan salinan dari server; jangan menjalankan ulang inference hanya karena file tidak ada di laptop.

- [Catatan pengembangan historis](DEVELOPMENT.md): runtime dan command stage terpisah.

Jika dokumen berbeda: cara pakai mengikuti ARCHICON/WORKFLOW, teknologi mengikuti TECHSTACK dan konfigurasi/source yang dirujuknya, prioritas mengikuti ROADMAP, hasil eksperimen mengikuti manifest/PROGRESS bertanggal. Snapshot lama tidak membuktikan status resource atau job saat ini.

Publikasi source19Oktober2026: host/path mesin pada dokumentasi adalah contoh yang disanitasi. Rencana kerja dan bukti berlabel lokal tetap pada workspace pemilik, tidak ikut GitHub. State checkpoint: CHANGELOG.md.
