import torch
import torch.nn as nn
import torchvision.models as models

class CNNFromScratch(nn.Module):
    """
    Kiến trúc CNN đơn giản (from scratch) để hiểu cách hoạt động của tích chập.
    """
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super(CNNFromScratch, self).__init__()
        self.features = nn.Sequential(
            # Lớp tích chập 1: Giữ cấu trúc không gian và học đặc trưng cục bộ
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Giảm kích thước ảnh, tăng receptive field
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name: str, num_classes: int, train_mode: str, dropout: float = 0.3):
    """
    Hàm khởi tạo mô hình dựa trên tham số truyền vào.
    """
    if model_name == 'cnn_small':
        return CNNFromScratch(num_classes, dropout)
    
    elif model_name == 'resnet18':
        # Tải mô hình ResNet18 đã được huấn luyện sẵn (Pretrained)
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        if train_mode == 'transfer':
            # Đóng băng (Freeze) toàn bộ backbone để chỉ học classifier head[cite: 1]
            for param in model.parameters():
                param.requires_grad = False
        
        # Thay thế lớp cuối cùng (FC layer) cho phù hợp với 6 lớp lỗi bề mặt thép[cite: 1]
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        return model
    
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}")