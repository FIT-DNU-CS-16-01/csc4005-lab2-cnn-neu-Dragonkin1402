
1. Hướng dẫn chạy chương trình

Để tái lập kết quả, hãy sử dụng các lệnh sau trong Terminal:

Huấn luyện mô hình CNN from scratch:

Bash
python -m src.train --data_dir NEU-CLS/train --model_name cnn_small --train_mode scratch --lr 0.001 --epochs 06 --use_wandb

Huấn luyện mô hình Transfer Learning (ResNet18):

Bash
python -m src.train --data_dir NEU-CLS/train --model_name resnet18 --train_mode finetune --lr 0.00005 --epochs 06 --augment --use_wandb
