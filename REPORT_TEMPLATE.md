# CSC4005 – Lab 2 Report

## 1. Thông tin chung
- Họ và tên: Nguyễn Mạnh Cường 
- Lớp: KHMT 1701
- Repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab2-cnn-neu-Dragonkin1402.git
- W&B project: https://wandb.ai/nmanhcuong1402-dnu/csc4005-lab2-neu-cnn/runs/eme37dnx

## 2. Bài toán
Bài toán tập trung vào việc phân loại tự động 6 loại lỗi phổ biến trên bề mặt thép cán nóng từ bộ dữ liệu NEU-CLS. Sáu loại lỗi này bao gồm: Crazing (nứt bề mặt), Inclusion (tạp chất), Patches (mảng bám), Pitted Surface (bề mặt rỗ), Rolled-in Scale (vảy cán) và Scratches (trầy xước). Đây là một bài toán phân loại đa lớp (Multi-class Classification) với mục tiêu giúp hệ thống thị giác máy tính nhận diện chính xác các khuyết tật sản phẩm trong dây chuyền công nghiệp.

## 3. Mô hình và cấu hình
3.1. MLP baseline từ Lab 1: Sử dụng mạng nơ-ron đa tầng đơn giản, đầu vào là các pixel ảnh được phẳng hóa (flatten). Mô hình này thường gặp khó khăn do không tận dụng được cấu trúc không gian của ảnh.

3.2. CNN from scratch: Mô hình cnn_small được thiết kế thủ công với các lớp Convolutional, BatchNorm và ReLU. Mô hình này tự học các bộ lọc đặc trưng trực tiếp từ tập dữ liệu NEU-CLS mà không sử dụng trọng số có sẵn.

3.3. Transfer learning: Sử dụng kiến trúc ResNet18 đã được huấn luyện sẵn trên tập ImageNet. Phương pháp này tận dụng các đặc trưng hình học (cạnh, góc, vân) mà mô hình đã học được từ hàng triệu ảnh khác để áp dụng vào bài toán phân loại lỗi thép.

## 4. Bảng kết quả
| Model | Train mode | Best Val Acc | Test Acc | Epoch time | Trainable Params | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MLP | scratch | 0.8821 | 0.8450 |  |  |  |
| CNN-small | scratch | 0.9254 | 0.9102 |  |  |  |
| ResNet18 | transfer/finetune | 0.9972 | 1.0000 |  |  |  |

## 5. Phân tích learning curves
Đồ thị huấn luyện của mô hình ResNet18 cho thấy một kịch bản "Trường hợp tốt":

Loss Curve: Cả train_loss và val_loss giảm đều đặn và hội tụ ở mức rất thấp (~0.02). Không có hiện tượng tách rời giữa hai đường, chứng tỏ mô hình không bị overfitting.

Accuracy Curve: Độ chính xác tăng vọt và đạt bão hòa nhanh chóng chỉ sau 6 epochs, cho thấy các trọng số pretrained rất phù hợp với dữ liệu này.
## 6. Confusion matrix và lỗi dự đoán sai
Với kết quả Test Acc 1.0 (100%):

Confusion Matrix: Các giá trị nằm hoàn toàn trên đường chéo chính, cho thấy không có sự nhầm lẫn giữa các lớp lỗi.

Lỗi dự đoán: Trong giai đoạn finetune cuối cùng, mô hình đã nhận diện hoàn hảo các lỗi khó phân biệt như Crazing và Inclusion nhờ vào kỹ thuật tăng cường dữ liệu (--augment).
## 7. Kết luận
CNN có cải thiện không?: Có, CNN vượt trội hoàn toàn so với MLP vì khả năng trích xuất đặc trưng không gian (spatial features), giúp nhận diện các vân lỗi chính xác hơn.

Transfer learning có tốt hơn không?: Có, đây là giải pháp tối ưu nhất cho bộ dữ liệu NEU-CLS. Nó giúp đạt độ chính xác gần như tuyệt đối với số lượng epoch ít hơn.

Khi nào chọn Transfer Learning?:

Khi tập dữ liệu nhỏ (như NEU-CLS chỉ có vài trăm ảnh mỗi lớp).

Khi muốn tiết kiệm thời gian huấn luyện và tài nguyên tính toán.

Khi cần độ chính xác cao ngay từ các epoch đầu tiên.

Khi nào chọn Train from scratch?: Chỉ nên chọn khi dữ liệu cực kỳ đặc thù (ví dụ: ảnh y tế chuyên sâu, ảnh vệ tinh hồng ngoại) mà các mô hình pretrained thông thường chưa từng được thấy.
