
1. Hướng dẫn chạy chương trình

Cấu trúc dự án: 
.
├── 📁 NEU-CLS/                  # Dữ liệu hình ảnh
│   └── 📁 train/                # Dữ liệu đã gộp để huấn luyện
│       ├── 📁 Crazing/          # Nứt bề mặt
│       ├── 📁 Inclusion/        # Tạp chất
│       ├── 📁 Patches/          # Mảng bám
│       ├── 📁 Pitted_Surface/   # Bề mặt rỗ
│       ├── 📁 Rolled-in_Scale/  # Vảy cán
│       └── 📁 Scratches/        # Trầy xước
├── 📁 src/                      # Mã nguồn (train.py, dataset.py, model.py)
├── 📁 outputs/                  # Lưu trữ Model Checkpoints
├── 📄 requirements.txt          # Thư viện cần thiết
└── 📄 README.md                 # Tài liệu hướng dẫn

Để tái lập kết quả, hãy sử dụng các lệnh sau trong Terminal:

Huấn luyện mô hình CNN from scratch:

Bash
python -m src.train --data_dir NEU-CLS/train --model_name cnn_small --train_mode scratch --lr 0.001 --epochs 06 --use_wandb

Huấn luyện mô hình Transfer Learning (ResNet18):

Bash
python -m src.train --data_dir NEU-CLS/train --model_name resnet18 --train_mode finetune --lr 0.00005 --epochs 06 --augment --use_wandb
