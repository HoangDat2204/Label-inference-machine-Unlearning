import os
import torch
import numpy as np
import argparse
import recovery as rs  # Import thư viện recovery như trong file gốc
from torchvision import transforms

def evaluate(args):
    # 1. Setup hệ thống (GPU/CPU)
    # Sử dụng hàm setup của thư viện recovery để tương thích device
    setup = rs.utils.system_startup()
    print(f"Device being used: {setup['device']}")

    # 2. Cấu hình Strategy (để khởi tạo dataloader đúng cách)
    # Các tham số này cần khớp với lúc train để Dataloader trả về đúng định dạng
    defs = rs.training_strategy('conservative')
    defs.batch_size = 128
    defs.augmentations = False # Đánh giá không cần augmentation

    # 3. Load Dataloader & Normalization Stats
    print(f"Loading dataset: {args.dataset}...")
    
    # Lưu ý: file gốc set normalize=False và tự normalize thủ công sau đó
    # excluded_num cần thiết lập để hàm construct_dataloaders chạy đúng logic chia tập tin
    excluded_num = 10000 if 'cifar' in args.dataset else 1000
    
    loss_fn, _tl, validloader, num_classes, _exd, dmlist, dslist = rs.construct_dataloaders(
        args.dataset.lower(), 
        defs, 
        data_path=f'datasets/{args.dataset.lower()}', 
        normalize=False, 
        exclude_num=excluded_num
    )

    # Tạo tensor mean/std để normalize thủ công (vì load với normalize=False)
    # Shape: [1, 3, 1, 1] để broadcast cộng trừ với batch ảnh
    dm = torch.as_tensor(dmlist, **setup).view(1, 3, 1, 1)
    ds = torch.as_tensor(dslist, **setup).view(1, 3, 1, 1)

    # 4. Khởi tạo Model Architecture
    print(f"Constructing model: {args.model}...")
    model, _ = rs.construct_model(args.model, num_classes=num_classes, num_channels=3)
    model.to(**setup)

    # 5. Load Weights (State Dict)
    print(f"Loading weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Không tìm thấy file model tại {args.model_path}")

    # Load file checkpoint
    # File gốc lưu dạng: {'net_sd': state_dict, ...} hoặc torch.save(model.state_dict())
    checkpoint = torch.load(args.model_path, map_location=setup['device'], weights_only = False)

    try:
        if isinstance(checkpoint, dict):
            if 'net_sd' in checkpoint:
                # Trường hợp file 'final.pth' trong code gốc
                model.load_state_dict(checkpoint['net_sd'])
                print("Loaded state_dict from key 'net_sd'")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("Loaded state_dict from key 'state_dict'")
            elif 'x_optimal' in checkpoint:
                 # Trường hợp file kết quả tái tạo (nếu muốn test model tái tạo)
                 # Thường x_optimal là ảnh, không phải weights, nhưng check cho chắc
                 print("Warning: File này có vẻ là kết quả tái tạo ảnh, không phải model weights.")
                 return
            else:
                # Trường hợp dict chính là state_dict
                model.load_state_dict(checkpoint)
                print("Loaded dictionary as state_dict")
        else:
            # Trường hợp save trực tiếp model (ít dùng) hoặc state_dict thuần
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Vui lòng kiểm tra lại cấu trúc file .pth")
        return

    # 6. Quá trình Evaluation
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Class-wise accuracy containers
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(**setup), targets.to(setup['device'])
            
            # Normalize thủ công: (Image - Mean) / Std
            inputs = (inputs - dm) / ds

            outputs = model(inputs)
            loss_output = loss_fn(outputs, targets)

            # --- SỬA LỖI TẠI ĐÂY ---
            # Kiểm tra nếu loss_fn trả về tuple (ví dụ: (loss, acc)) thì lấy phần tử đầu tiên
            if isinstance(loss_output, tuple):
                loss = loss_output[0]
            else:
                loss = loss_output
            # -----------------------

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Tính accuracy cho từng class
            c = (predicted == targets).squeeze()
            
            # Xử lý trường hợp batch size = 1 hoặc c không có chiều (scalar)
            if c.ndim == 0:
                c = c.unsqueeze(0)
                
            for i in range(len(targets)):
                label = targets[i]
                # Đảm bảo index không vượt quá giới hạn (an toàn)
                if label < len(class_correct):
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(validloader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

    # 7. Kết quả cuối cùng
    final_acc = 100. * correct / total
    avg_loss = test_loss / len(validloader)
    
    print("-" * 50)
    print(f"Evaluation Results for {args.model} on {args.dataset}")
    print(f"Model Path: {args.model_path}")
    print("-" * 50)
    print(f"Final Accuracy: {final_acc:.2f}%")
    print(f"Average Loss:   {avg_loss:.4f}")
    print("-" * 50)
    
    # In Accuracy từng class (tùy chọn)
    print("Class-wise Accuracy:")
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if len(classes) != num_classes: # Fallback nếu không phải CIFAR10
        classes = [str(i) for i in range(num_classes)]
        
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'{classes[i]:<10}: {100 * class_correct[i] / class_total[i]:.2f}%')
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ResNet18 on CIFAR10')
    
    # Các tham số mặc định dựa trên file gốc
    parser.add_argument('--model', default='ResNet18', type=str, help='Tên model (ConvNet, ResNet18...)')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('--model_path', type=str, required=True, help='Đường dẫn đến file .pth cần test')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    
    # Set seed giống file gốc để đảm bảo tái lập
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    evaluate(args)