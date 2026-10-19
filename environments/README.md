# Environment

Laptop dan server memiliki `.venv` Python 3.12 yang terpisah dari proyek lama.
Dependency audit telah dipasang dan diuji pada kedua mesin: lihat
`audit.requirements.txt`. Server juga memakai `camera.requirements.txt` dengan
PyCOLMAP 4.2.1 CPU. Requirements berisi pin package langsung yang diuji;
inventory aktual setiap run dicatat oleh preflight.

```bash
rtk proxy .venv/bin/python -m pip install -r environments/audit.requirements.txt
# Hanya server, bila environment belum terpasang:
rtk proxy .venv/bin/python -m pip install -r environments/camera.requirements.txt
```

Runtime inference part2 sudah diuji pada DA3-LARGE empat view. Splatting belum diuji.
Runtime lama
PyTorch `2.11.0+cu130` mendukung `sm_120`, tetapi compiler sistem CUDA 12.0 tidak
dipakai membangun kernel Blackwell. Pilih stack inference dan compiler yang sesuai,
uji load snapshot DA3 lokal, lalu uji rasterizer forward/backward secara terpisah.
Pisahkan environment inference/geometry bila dependency bertentangan. Environment
lama dan shared model cache tetap dipakai read-only.

## Runtime inference yang digunakan

Server `.venv-inference` dibuat dengan menyalin 2,1 GiB `.venv` lama ke proyek part2.
Source DA3 disalin ke `vendor/depth_anything_3`; import ditegakkan ke source ini.
Tidak ada pemasangan package atau unduhan checkpoint baru. Gunakan interpreter
`.venv-inference/bin/python`, bukan pip/CLI script hasil salinan yang shebang-nya
masih menunjuk lokasi lama. Package sistem/user diwarisi: terutama PyTorch dari
`~/.local/lib/python3.12/site-packages`. Jadi ini isolasi environment CPU dan source,
bukan environment hermetik yang kebal perubahan package global.

Runtime aktual: PyTorch2.11.0+cu130, torchvision0.26.0, xformers0.0.35,
NumPy1.26.4, safetensors0.8.0, Hugging Face Hub2.2.0, Pillow12.3.0. Inventory seluruh
package dan hash93file vendor tercatat dalam `inference-inventory.json` dan
`inference-bootstrap.json` (artefak mesin, di-ignore). Exact checkpoint/source/API
hashes serta package versions disimpan di manifest/report smoke. CPU audit tetap
memakai NumPy2.5.4 di `.venv` terpisah.

Smoke offline, `infer_gs=False`, tanpa gsplat. Peringatan missing gsplat tidak
menghalangi depth. Model menghasilkan optical-axis Z280×504 pada input portrait;
API menyelaraskan scale dan mengganti output E/K dengan input conditioning.
Raw predicted E/K disimpan terpisah. Scale tetap relatif.

## Geometry backend yang diuji

`.venv-geometry` adalah salinan runtime inference dengan Open3D 0.19.0 dipasang
menggunakan `python -m pip install --no-deps --ignore-installed open3d==0.19.0`.
Metadata0.20 tersisa dari salinan diarsipkan hanya di environment baru; source/runtime
lama dan inference tidak diubah. Guard memeriksa versi distribusi dan module aktual.
Direct pins: `geometry.requirements.txt`; inventory: `geometry-inventory.json`.

Open3D 0.20 memiliki regresi ScalableTSDFVolume block activation float32 depth:
fixture plane menghasilkan volume kosong meskipun RGBD depth benar.0.19 lolos fixture
world-plane yang sama. Ini diatasi dengan pin backend, tanpa mengubah depth_scale1
atau unit scene. [Upstream issue](https://github.com/isl-org/Open3D/issues/7592).
Geometry memakai CPU,CUDA disembunyikan,empat thread,dan supervisor hostRSS6GiB.

Mask orang memakai TorchVision SSDLite COCO weight resmi13,4MiB yang diunduh ke
`models/masking/`,SHA256
`a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`.
Tidak ada model DA3 baru diunduh. Detektor melewatkan orang kecil/occluded; anotasi
manual per-frame ditambahkan dengan image-hash provenance.
