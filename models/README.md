# Model cache

Checkpoint tetap berada di cache server yang ditentukan profil mesin privat.
Contoh lokasi: `/srv/model-cache/huggingface/hub`.

Server memiliki tautan `models/huggingface` menuju cache tersebut. Tidak ada bobot
yang disalin ke laptop atau diunduh ulang saat scaffold.

Registry model ada di `configs/models.yaml`. Pilih exact snapshot revision untuk
reproduksi. Adapter memperlakukan cache sebagai sumber read-only;
tautan symlink sendiri tidak menegakkan proteksi tulis filesystem.

MapAnything belum dipasang atau diunduh pada tahap ini.
