
1. Hướng dẫn chạy chương trình

Cấu trúc dự án: 
.
├── NEU-CLS/                # Thư mục gốc chứa dữ liệu
│   └── train/              # Thư mục chứa dữ liệu huấn luyện đã gộp
│       ├── Crazing/        # Chứa ảnh lỗi nứt bề mặt
│       ├── Inclusion/      # Chứa ảnh lỗi tạp chất
│       ├── Patches/        # Chứa ảnh lỗi mảng bám
│       ├── Pitted_Surface/ # Chứa ảnh lỗi bề mặt rỗ
│       ├── Rolled-in_Scale/# Chứa ảnh lỗi vảy cán
│       └── Scratches/      # Chứa ảnh lỗi trầy xước
├── src/                    # Mã nguồn dự án (train.py, dataset.py,...)
├── configs/                # Các file cấu hình huấn luyện
├── docs/                   # Tài liệu hướng dẫn
├── outputs/                # Nơi lưu trữ kết quả, model checkpoints
├── wandb/                  # Logs huấn luyện của Weights & Biases
├── requirements.txt        # Danh sách các thư viện cần cài đặt
├── README.md               # File hướng dẫn chạy và báo cáo
└── .venv/                  # Môi trường ảo Python

Để tái lập kết quả, hãy sử dụng các lệnh sau trong Terminal:

Huấn luyện mô hình CNN from scratch:

Bash
python -m src.train --data_dir NEU-CLS/train --model_name cnn_small --train_mode scratch --lr 0.001 --epochs 06 --use_wandb

Huấn luyện mô hình Transfer Learning (ResNet18):

Bash
python -m src.train --data_dir NEU-CLS/train --model_name resnet18 --train_mode finetune --lr 0.00005 --epochs 06 --augment --use_wandb
