# Room tour — ArchiCon

**16 Oktober 2026.** Viewer HTML memakai identitas ArchiCon dengan desain workspace arsitektur/interior: Carbon Gray100, IBM Plex Sans, garis tipis, kontrol siku dan aksen IBM Blue. Acuan: [DESIGN.md](../DESIGN.md). Dashboard dan viewer memakai warna/font yang sama; denah ditampilkan pada lembar gambar terang di dalam aplikasi dark mode.

## Buka hasil

Pada dashboard, pilih job → **Hasil akhir** → **Buka HTML** untuk area tampilan lebih luas. Viewer juga dapat dibuka langsung dari `end_product/JOB/vNNN/room_tour/index.html` tanpa backend atau internet. Font, mesh viewer, bukti denah, data kamera, JavaScript dan CSS ditanam di HTML; tidak ada CDN atau font jaringan.

Hasil desain terbaru: lantai3_part1_1016_1525/v002 (`../end_product/lantai3_part1_1016_1525/v002/room_tour/index.html`, lokal). Dashboard dan latest.json menunjuk v002. Versi v001 dan produk lama tetap utuh; desain baru menjadi default untuk HTML yang dibangun pipeline berikutnya.

## Kontrol ruang

- Desktop: navigasi komponen di kiri menampilkan nama, thumbnail denah asli dan jumlah kamera komponen. Mobile: gunakan dropdown komponen di atas viewer. Pemilihan selalu tersinkron.
- **Tur 3D**: geser untuk orbit, roda/cubit untuk zoom, klik kanan untuk pan. **Ikhtisar** mengembalikan tampilan keseluruhan komponen. Potongan atap membuka bagian atas model saat ikhtisar; kontrol dinonaktifkan saat memakai kamera rekaman atau denah.
- **Putar tur** dan slider kamera mengikuti pose/timestamp rekaman. Tur berhenti pada jeda sumber; lanjutkan dengan Lewati jeda & putar. Saat berada pada kamera rekaman, geser/tombol panah untuk melihat sekitar dari posisi yang sama.
- **Denah 2D**: bukti irisan, garis geometri, lintasan dan kamera ditampilkan pada lembar gambar. Kamera pada denah dapat dipilih untuk membuka sudut rekamannya. Tombol Bukti irisan mengatur overlay; Unduh SVG menyimpan gambar dan metadata kalibrasi draft.
- Klik dua titik pada denah untuk mengukur proyeksi X/Z. Masukkan panjang nyata segmen tersebut untuk kalibrasi draft. Simpan JSON sebelum menutup HTML; muat kembali hanya pada mesh dengan hash yang sama. Draft setiap komponen terpisah.
- **Info sumber** memuat asal data dan batas rekonstruksi. Navigasi tab mendukung tombol panah kiri/kanan serta Home/End, dengan indikator fokus keyboard.

Jumlah sudut rekaman pada header menghitung frame-ID unik lintas-komponen; jumlah kamera pada masing-masing komponen dapat beririsan. Bentang geometri tetap diberi satuan u sampai ada acuan nyata. Komponen lokal memiliki world independen; denah merupakan irisan geometri, belum denah arsitektur semantik, dan estimasi meter belum hasil survei.

Pada job dengan S06G aktif dan alignment yang lolos, pilihan **Arsitektur gabungan · kandidat** muncul di awal daftar. Ini menampilkan satu world gabungan, dengan denah dan kamera yang ditransformasikan ke world root yang sama. World yang sudah digabung tidak diduplikasi di sidebar; GLB lokal aslinya tetap tersedia di folder `mesh/`. Semua world fallback tetap ditampilkan. Jika world utama belum terhubung, grup lain yang terbukti tersambung dapat menjadi kandidat; ini tetap bagian bangunan, bukan seluruh bangunan. Lihat `reports/global-alignment.json` untuk membership/root dan fallback. Lubang dan permukaan ganda masih mungkin ada. Bila status `no_alignment`, pilihan gabungan serta `mesh/architecture.glb` tidak dibuat. Hasil lama tidak berubah otomatis.

## Implementasi dan bukti

Template/styles/interaksi: `web/tour/`. `web/build.mjs` menyematkan font dari `web/dashboard/fonts/` beserta lisensi OFL; Three.js dan esbuild tetap pada pin proyek. Perubahan ini hanya presentasi dan packaging, tanpa inference baru atau layanan review agent.

v002 memiliki mesh/PNG/SVG dan embedded data kamera/mesh/denah identik dengan v001. Keputusan kualitas tetap pending/unassessed; paket review baru terikat v002 dan bundle versi tersebut. Metadata durasi rekonstruksi serta packet review v001 dipertahankan.

Bukti di `outputs/jobs/lantai3_part1_1016_1525/qa/tour-redesign/`: `delivery-browser-qa-report.json`, `product-verification.json`, `delivery-packaging.json`, screenshot desktop/mobile/denah, serta ekspor kalibrasi/SVG yang diuji. QA pipeline ada di `qa/v002/`. Offline browser QA mencakup seluruh4komponen,252frame-ID unik, playback/gap/lookaround, keyboard, pengukuran/kalibrasi/JSON/SVG dan lebar320/390/768/1280/1440px, tanpa pageerror atau horizontal overflow.
