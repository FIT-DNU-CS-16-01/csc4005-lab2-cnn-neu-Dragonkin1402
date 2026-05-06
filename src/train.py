from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import create_dataloaders
# Lưu ý: Đảm bảo src/model.py của bạn có hàm get_model
from src.model import get_model 
from src.utils import (
    EarlyStopping,
    classification_report_dict,
    compute_accuracy,
    ensure_dir,
    plot_curves,
    save_confusion_matrix,
    save_history_csv,
    save_json,
    set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CNN Training for NEU Surface Defect Classification')
    # Các tham số bắt buộc và thông tin project
    parser.add_argument('--data_dir', type=str, required=True, help='Đường dẫn tới NEU-CLS.zip hoặc thư mục dữ liệu')
    parser.add_argument('--project', type=str, default='csc4005-lab2-neu-cnn')
    parser.add_argument('--run_name', type=str, default='cnn_run')
    
    # Tham số quan trọng của Lab 2
    parser.add_argument('--model_name', type=str, default='cnn_small', help='cnn_small, resnet18, resnet50...')
    parser.add_argument('--train_mode', type=str, choices=['scratch', 'transfer', 'finetune'], default='scratch')
    
    # Hyperparameters
    parser.add_argument('--optimizer', type=str, choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument('--scheduler', type=str, choices=['none', 'plateau'], default='none')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true')
    return parser.parse_args()


def get_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    if name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    start_time = time.time()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        
    epoch_time = time.time() - start_time
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred), epoch_time


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred), y_true, y_pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = ensure_dir(Path('outputs') / args.run_name)

    # Khởi tạo DataLoader
    data = create_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        augment=args.augment,
        num_workers=args.num_workers,
    )

    # Khởi tạo CNN Model thay vì MLP
    model = get_model(
        model_name=args.model_name,
        num_classes=len(data.class_names),
        train_mode=args.train_mode,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) if args.scheduler == 'plateau' else None

    use_wandb = args.use_wandb and wandb is not None
    if use_wandb:
        wandb.init(project=args.project, name=args.run_name, config=vars(args))

    history: list[dict[str, float]] = []
    early_stopper = EarlyStopping(patience=args.patience)
    best_val_acc = 0.0
    
    print(f"🚀 Bắt đầu huấn luyện: {args.model_name} ({args.train_mode})")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, epoch_time = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, data.val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step(val_loss)
            
        lr_current = optimizer.param_groups[0]['lr']
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr_current,
            'epoch_time_sec': epoch_time # Quan trọng để so sánh Performance
        }
        history.append(row)
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {train_acc:.4f}/{val_acc:.4f} | Time: {epoch_time:.2f}s")
        
        if use_wandb:
            wandb.log(row)
            
        if early_stopper.step(val_loss):
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            
        if early_stopper.should_stop:
            print(f"🛑 Early stopping tại epoch {epoch}")
            break

    # Đánh giá cuối cùng trên Test set
    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, data.test_loader, criterion, device)
    
    # Lưu kết quả
    report = classification_report_dict(y_true, y_pred, data.class_names)
    save_confusion_matrix(y_true, y_pred, data.class_names, output_dir / 'confusion_matrix.png')
    plot_curves(history, output_dir / 'curves.png')
    save_history_csv(history, output_dir / 'history.csv')
    
    metrics = {
        'model_name': args.model_name,
        'train_mode': args.train_mode,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'classification_report': report
    }
    save_json(metrics, output_dir / 'metrics.json')

    if use_wandb:
        wandb.log({'test_acc': test_acc})
        wandb.finish()

if __name__ == '__main__':
    main()