import os
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import recovery as rs
import argparse
from PIL import Image
import glob
from scipy.optimize import nnls
from sklearn.linear_model import Lasso


recons_config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.04, 
              optim='adamw',
              restarts=5,
              max_iterations=10000,
              total_variation=1e-2,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

def new_plot(tensor, title="", path=None):
    if tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(2 * tensor.shape[0], 3))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())
    plt.title(title)
    plt.savefig(path)

def process_recons_results(result, ground_truth, figpath, recons_path, filename):
    output_list, stats, history_list, x_optimal = result
    x_optimal = x_optimal.detach().cpu()
    test_mse = (x_optimal - ground_truth.cpu()).pow(2).mean()
    test_psnr = rs.metrics.psnr(x_optimal, ground_truth, factor=1/ds)
    title = f"MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | "
    new_plot(torch.cat([ground_truth, x_optimal]), title, path=os.path.join(figpath, f'{filename}.png'))
    torch.save({'output_list': output_list.cpu(), 'stats': stats, 'history_list': history_list, 'x_optimal': x_optimal}, open(os.path.join(recons_path, f'{filename}.pth'), 'wb'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple argparse.')
    parser.add_argument('--model', default='ConvNet', type=str, help='Vision model.')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--ft_samples', default=32, type=int)
    parser.add_argument('--unlearn_samples', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int, help='updated epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model_save_folder', default='results/models', type=str, help='folder of pretrained models')

    args = parser.parse_args()

    print(args.__dict__)

    img_size = 32 if 'cifar' in args.dataset else 96
    excluded_num = 10000 if 'cifar' in args.dataset else 1000
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    
    load_folder_name = f'{args.model.lower()}_{args.dataset.lower()}_ex{excluded_num}_s0'
    save_folder_name = f'ex{args.ft_samples}_un{args.unlearn_samples}_ep{args.epochs}_seed{args.seed}'
    save_folder = os.path.join(args.model_save_folder, load_folder_name, save_folder_name)
    os.makedirs(save_folder, exist_ok=True)
    
    final_dict = torch.load(os.path.join(args.model_save_folder, load_folder_name, 'final.pth'), weights_only = False)
    setup = rs.utils.system_startup()
    defs = rs.training_strategy('conservative')
    defs.lr = args.lr
    defs.epochs = args.epochs
    defs.batch_size = 128
    defs.optimizer = 'SGD'
    defs.scheduler = 'linear'
    defs.warmup = False
    defs.weight_decay  = 0.0
    defs.dropout = 0.0
    defs.augmentations = False
    defs.dryrun = False

    
    loss_fn, _tl, validloader, num_classes, _exd, dmlist, dslist =  rs.construct_dataloaders(args.dataset.lower(), defs, data_path=f'datasets/{args.dataset.lower()}', normalize=False, exclude_num=excluded_num)
    dm = torch.as_tensor(dmlist, **setup)[:, None, None]
    ds = torch.as_tensor(dslist, **setup)[:, None, None]
    normalizer = transforms.Normalize(dmlist, dslist)


    # *** used for batch case ***
    excluded_data = final_dict['excluded_data']
    index = torch.tensor(np.random.choice(len(excluded_data[0]), args.ft_samples, replace=False))
    print("Batch index", index.tolist())
    X_all, y_all = excluded_data[0][index], excluded_data[1][index]
    print("FT data size", X_all.shape, y_all.shape)
    trainset_all = rs.data_processing.SubTrainDataset(X_all, y_all, transform=transforms.Normalize(dmlist, dslist))
    trainloader_all = torch.utils.data.DataLoader(trainset_all, batch_size=min(defs.batch_size, len(trainset_all)), shuffle=True,  num_workers=8, pin_memory=True)
    

    ## load state dict
    state_dict =  final_dict['net_sd']


    model_pretrain, _ = rs.construct_model(args.model, num_classes=num_classes, num_channels=3)
    model_pretrain.load_state_dict(state_dict)
    model_pretrain.eval()
    
    
    model_ft, _ = rs.construct_model(args.model, num_classes=num_classes, num_channels=3)
    model_ft.load_state_dict(state_dict)
    model_ft.eval()


    print("Train full model.")
    ft_folder = os.path.join(save_folder, 'full_ft')
    os.makedirs(ft_folder, exist_ok=True)
    model_ft.to(**setup)
    ft_stats = rs.train(model_ft, loss_fn, trainloader_all, validloader, defs, setup=setup, ckpt_path=ft_folder, finetune=True)
    model_ft.cpu()
    resdict = {'tr_args': args.__dict__,
        'tr_strat': defs.__dict__,
        'stats': ft_stats,
        'batch_index': index,
        'train_data': (X_all, y_all)}
    torch.save(resdict, os.path.join(ft_folder, 'finetune_params.pth'))
    ft_diffs = [(ft_param.detach().cpu() - org_param.detach().cpu()).detach() for (ft_param, org_param) in zip(model_ft.parameters(), model_pretrain.parameters())]

    # --------------------------------------------------------------------------
    # HÀM TIỆN ÍCH TÍNH TOÁN 
    # --------------------------------------------------------------------------
    def normalize_to_unit(v):
        """
        Chuẩn hóa vector v về độ dài bằng 1 (Unit Vector).
        Input: Numpy array 1D.
        Output: Numpy array 1D có L2 norm = 1.
        """
        norm = np.linalg.norm(v)
        # Thêm epsilon nhỏ xíu để tránh lỗi chia cho 0 nếu vector toàn số 0
        if norm < 1e-12: 
            return v
        return v / norm

    def flatten_gradients(gradient_list):
        """
        Input: List các tensor (weights, bias của từng layer).
        Output: Một Tensor 1 chiều duy nhất (Vector khổng lồ).
        """
        # Gộp tất cả tensor lại thành 1 chuỗi vector dài và chuyển về CPU để giải phương trình
        return torch.cat([p.flatten().detach().cpu() for p in gradient_list])

    def predict_label_distribution_corrected(approx_diff, representative_gradients, batch_size):
        # 1. Tính Norm thực tế của từng Class từ Gradient đại diện
        # (Giả sử representative_gradients chưa bị normalize ở bước lưu file)
        basis_norms = []
        basis_vectors = []
        
        for g in representative_gradients:
            grad_flat = g[-1].detach().cpu().numpy().flatten()
            norm = np.linalg.norm(grad_flat)
            basis_norms.append(norm + 1e-9) # Tránh chia cho 0
            basis_vectors.append(grad_flat / (norm + 1e-9)) # Chuẩn hóa Basis về 1
            
        A = np.stack(basis_vectors, axis=1) # Ma trận các vector đơn vị
        
        # 2. Target cũng chuẩn hóa về 1 để so sánh góc
        target_raw = approx_diff[-1].detach().cpu().numpy().flatten()
        target_norm = np.linalg.norm(target_raw)
        b = target_raw / (target_norm + 1e-9)
        
        # 3. Giải NNLS để tìm đóng góp về HƯỚNG
        coeffs_direction, _ = nnls(A, b)
        
        # 4. [BƯỚC SỬA LỖI] Hiệu chỉnh lại bằng độ dài gốc
        # Logic: Contribution_Angle ≈ Count * Original_Norm
        # => Count ≈ Contribution_Angle / Original_Norm
        # Tuy nhiên, cần nhân lại với target_norm để khôi phục scale (tùy chọn, nhưng chia tỷ lệ là quan trọng nhất)
        
        # Cách đơn giản nhất: Phạt những thằng có Norm quá to (vì nó chiếm hướng dễ quá)
        estimated_counts_raw = coeffs_direction / np.array(basis_norms)
        
        # 5. Làm tròn
        final_counts = prepare_and_round(estimated_counts_raw, batch_size)
        
        return final_counts

    def round_preserving_sum(weights, target_sum):
        """
        Làm tròn các số thực trong 'weights' sao cho tổng của chúng bằng 'target_sum'.
        Sử dụng phương pháp Largest Remainder Method.
        """
        # 1. Lấy phần nguyên (Floor)
        floored_weights = np.floor(weights).astype(int)
        
        # 2. Tính tổng hiện tại và phần thiếu
        current_sum = np.sum(floored_weights)
        remainder = target_sum - current_sum
        
        # 3. Tính phần thập phân dư ra (để biết ưu tiên cộng thêm cho ai)
        decimal_parts = weights - floored_weights
        
        # 4. Sắp xếp giảm dần dựa trên phần dư (ai dư nhiều nhất thì được ưu tiên cộng 1)
        # argsort trả về index tăng dần -> [::-1] đảo ngược để thành giảm dần
        sort_indices = np.argsort(decimal_parts)[::-1]
        
        # 5. Cộng bù 1 đơn vị vào các phần tử có phần dư lớn nhất cho đến khi đủ tổng
        for i in range(remainder):
            idx = sort_indices[i]
            floored_weights[idx] += 1
            
        return floored_weights
    
    def predict_with_lasso(approx_diff, representative_gradients, batch_size):
        # Chuẩn bị dữ liệu (như cũ)
        target = normalize_to_unit(approx_diff[-1].detach().cpu().numpy().flatten())
        A = np.stack([normalize_to_unit(g[-1].detach().cpu().numpy().flatten()) for g in representative_gradients], axis=1)
        
        # Cấu hình Lasso: positive=True (tương đương NNLS)
        # alpha: Hệ số phạt, càng lớn càng ép nhiều số về 0 (cần tinh chỉnh, vd: 0.001 -> 0)
        lasso = Lasso(alpha=0.01, positive=True, fit_intercept=False)
        lasso.fit(A, target)
        
        return prepare_and_round(lasso.coef_, batch_size)
    
    def predict_label_distribution_bias_inv(approx_diff, representative_gradients, batch_size):
        """
        Dự đoán dùng phương pháp nghịch đảo ma trận: x = A^(-1) * b
        Chỉ áp dụng cho Bias lớp cuối (tạo ra ma trận vuông 10x10).
        """
        
        # 1. Chuẩn bị Vector b (Target)
        target_raw = approx_diff[-1].detach().cpu().numpy().flatten()
        b = target_raw
        
        # 2. Chuẩn bị Ma trận A (Basis)
        A_columns = []
        for g in representative_gradients:
            bias_grad = g[-1].detach().cpu().numpy().flatten()
            bias_grad_norm = bias_grad
            A_columns.append(bias_grad_norm)
            
        # Shape: (10, 10) - 10 hàng (bias features), 10 cột (classes)
        A = np.stack(A_columns, axis=1)
        
        # 3. Tính toán x = A_inv * b
        try:
            # Tính ma trận nghịch đảo
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            print("[WARNING] Ma trận bị suy biến (Singular), không thể nghịch đảo trực tiếp!")
            print("--> Đang áp dụng Regularization (thêm nhiễu vào đường chéo)...")
            
            # Kỹ thuật: Cộng thêm 1e-6 vào đường chéo để ma trận khả nghịch
            epsilon = 1e-6
            A_safe = A + np.eye(A.shape[0]) * epsilon
            A_inv = np.linalg.inv(A_safe)
            
        # Nhân ma trận: x = A^-1 . b
        coefficients = np.dot(A_inv, b)
        
        # 4. Làm tròn kết quả
        final_counts = prepare_and_round(coefficients, batch_size)
        
        return final_counts

    def predict_label_distribution_bias_only_normalized(approx_diff, representative_gradients, batch_size, print_0 = True):
        """
        Dự đoán phân phối nhãn dùng Bias Gradient + Chuẩn hóa + Non-negative Least Squares (NNLS).
        """
        
        # 1. Trích xuất Bias Gradient của Delta W (Target)
        target_raw = approx_diff[-1].detach().cpu().numpy().flatten()
        # target_raw = approx_diff
        # b = normalize_to_unit(target_raw) # Chuẩn hóa Target
        b = target_raw

        # 2. Trích xuất và Chuẩn hóa cơ sở (Basis)
        A_columns = []
        for g in representative_gradients:
            bias_grad = g[-1].detach().cpu().numpy().flatten()
            bias_grad_norm = bias_grad # Chuẩn hóa Basis
            A_columns.append(bias_grad_norm)
            
        # Tạo ma trận A (Shape: 10x10)
        A = np.stack(A_columns, axis=1)
        
        # 3. GIẢI HỆ PHƯƠNG TRÌNH VỚI RÀNG BUỘC KHÔNG ÂM (NNLS)
        # Hàm nnls trả về (solution_vector, residual)
        # coefficients đảm bảo luôn >= 0
        coefficients, residual = nnls(A, b) 
        if (print_0):
            print(A)
            print(b)
            print(coefficients)
            print(residual)
        # 4. Tính toán số lượng (Scaling lại theo batch_size)
        # Vì coefficients đã không âm rồi, hàm prepare_and_round chỉ cần lo việc làm tròn thôi
        final_counts = prepare_and_round(coefficients, batch_size)
        
        return final_counts

    def predict_label_distribution_bias_only(approx_diff, representative_gradients, batch_size):
        """
        Dự đoán phân phối nhãn chỉ dựa trên Gradient của Bias lớp cuối cùng (fc.bias).
        Nhanh hơn và đôi khi ổn định hơn so với dùng toàn bộ weights.
        
        Args:
            approx_diff: List chứa Gradient của batch cần unlearn (Delta W).
            representative_gradients: List chứa 10 bộ Gradient đại diện (Basis).
            batch_size: Số lượng ảnh trong batch.
        """
        
        # 1. Trích xuất Gradient Bias lớp cuối (Phần tử cuối cùng trong list)
        # Đối với ResNet18, approx_diff[-1] chính là gradient của fc.bias (shape: [10])
        target_bias_grad = approx_diff[-1].detach().cpu().numpy().flatten()
        
        # Kiểm tra nhanh kích thước để đảm bảo đúng là bias (CIFAR10 thì phải là 10)
        if len(target_bias_grad) != 10:
            print(f"[WARNING] Gradient cuối cùng có kích thước {len(target_bias_grad)}, có thể không phải là Bias lớp fc (mong đợi 10)!")

        # 2. Trích xuất Bias Gradient từ bộ cơ sở (Representative Gradients)
        # representative_gradients là list 10 phần tử (tương ứng 10 class)
        # Mỗi phần tử là một list các tensor gradient toàn mạng -> lấy [-1] của từng cái
        basis_bias_grads = [g[-1].detach().cpu().numpy().flatten() for g in representative_gradients]
        print(basis_bias_grads)
        # 3. Tạo Ma trận A (Shape: 10x10)
        # Mỗi cột là vector bias gradient của một class
        A = np.stack(basis_bias_grads, axis=1) 
        
        # Vector b (Shape: 10)
        b = target_bias_grad
        
        # 4. Giải hệ phương trình tuyến tính Ax = b
        # Vì ma trận nhỏ (10x10), việc tính toán diễn ra tức thì
        coefficients, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # 5. Làm tròn và chuẩn hóa số lượng
        final_counts = prepare_and_round(coefficients, batch_size)
        
        return final_counts

    def predict_label_distribution(approx_diff, representative_gradients, batch_size):
        """
        approx_diff: Gradient quan sát được (Delta W) - List of Tensors
        representative_gradients: List chứa 10 bộ Gradient đại diện cho 10 class (đã tính từ Probing Samples)
        batch_size: Số lượng ảnh trong batch cần dự đoán
        """
        
        # Bước 1: Flatten các dữ liệu đầu vào
        # Vector b (Target): approx_diff duỗi phẳng
        b = flatten_gradients(approx_diff).numpy() # Chuyển sang numpy để tính toán đại số
        
        # Ma trận A (Basis): Mỗi cột là một Gradient đại diện duỗi phẳng
        # representative_gradients là List[List[Tensor]], cần duỗi từng cái
        A_columns = [flatten_gradients(g).numpy() for g in representative_gradients]
        A = np.stack(A_columns, axis=1) # Shape: (Số tham số model, 10 class)
        
        # Bước 2: Giải hệ phương trình tuyến tính Ax = b (Tìm x sao cho sai số bé nhất)
        # Sử dụng Least Squares vì hệ phương trình này dư thừa (số tham số >>> số class)
        # rcond=None để dùng giá trị mặc định cho việc cắt bỏ các giá trị kỳ dị
        coefficients, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Bước 3: Xử lý hệ số (Code logic bạn đã cung cấp)
        # coefficients chính là weights thô
        final_counts = prepare_and_round(coefficients, batch_size)
        
        return final_counts

    def prepare_and_round(weights, batch_size):
        # Loại bỏ giá trị âm (Gradient noise có thể gây ra hệ số âm)
        weights = np.maximum(weights, 0)
        
        total = np.sum(weights)
        if total > 0:
            # Chuẩn hóa về đúng tỷ lệ batch_size
            weights = weights / total * batch_size
        else:
            # Trường hợp xấu: weights toàn 0 (hiếm gặp)
            weights = np.zeros_like(weights)
            
        return round_preserving_sum(weights, batch_size)
   
    def predict_label_distribution_weight_col_norm_sum(approx_diff, representative_gradients, batch_size, print_0 = True):
        """
        Dự đoán phân phối nhãn.
        Logic: 
        - Lấy ma trận Weights [10, 512].
        - Áp dụng normalize_to_unit cho từng cột trong 512 cột.
        - Cộng tổng các cột lại thành vector [10].
        - Giải NNLS.
        """

        # --- HELPER: Hàm xử lý vector dùng đúng hàm normalize_to_unit của bạn ---
        def aggregate_weight_numpy(grad_list):
            # Lấy Weight Tensor [10, 512] và chuyển sang Numpy
            # grad_list[-2] là weights lớp cuối (liền trước bias)
            weight_matrix = grad_list[-2].detach().cpu().numpy()
            
            # weight_matrix.shape[1] là 512 (số lượng feature/cột)
            num_features = weight_matrix.shape[1]
            
            # List chứa 512 vector đã chuẩn hóa
            normalized_cols = []
            
            # Duyệt qua từng cột (Feature Vector)
            for i in range(num_features):
                col_vector = weight_matrix[:, i] # Lấy vector cột thứ i (10 chiều)
                
                # [QUAN TRỌNG] Gọi hàm chuẩn hóa của bạn cho vector này
                norm_col = normalize_to_unit(col_vector)
                
                normalized_cols.append(norm_col)
                
            # Gộp lại thành ma trận [10, 512] đã chuẩn hóa
            normalized_matrix = np.stack(normalized_cols, axis=1)
            
            # Cộng tổng theo chiều ngang (axis=1) -> ra vector [10]
            aggregated_vector = np.sum(normalized_matrix, axis=1)
            
            return aggregated_vector

        # ---------------------------------------------------------
        
        # 1. Xử lý Delta W (Target)
        # Kết quả trả về là vector 10 chiều (tổng của 512 vector đơn vị)
        target_vector = aggregate_weight_numpy(approx_diff)
        
        # Chuẩn hóa vector tổng này lần cuối để so sánh hướng với Basis
        b = normalize_to_unit(target_vector)
        
        # 2. Xử lý Cơ sở (Basis)
        A_columns = []
        for g in representative_gradients:
            bias_grad = g[-1].detach().cpu().numpy().flatten()
            bias_grad_norm = normalize_to_unit(bias_grad) # Chuẩn hóa Basis
            A_columns.append(bias_grad_norm)
            
        # Tạo ma trận A (Shape: 10x10)
        A = np.stack(A_columns, axis=1)
        
        # 3. Giải NNLS
        coefficients, residual = nnls(A, b)
        if (print_0):
            print(A)
            print(b)
            print(coefficients)
            print(residual)
        # 4. Làm tròn & Scale về batch size
        # (Đảm bảo bạn đã có hàm prepare_and_round ở scope ngoài hoặc import vào)
        final_counts = prepare_and_round(coefficients, batch_size)
        
        return final_counts
   
    def predict_label_distribution_bias_peeling(approx_diff, representative_gradients, batch_size):
        """
        Dự đoán phân phối nhãn dùng Bias Gradient + Thuật toán Bóc tách (Greedy Peeling).
        
        Logic:
        1. Input: Vector Bias DeltaW (Target) và 10 Vector Bias Đại diện (Basis).
        2. Chuẩn hóa Basis về 1 (để so sánh hướng).
        3. KHÔNG chuẩn hóa Target (để giữ độ lớn mà bóc tách dần).
        4. Lặp 'batch_size' lần: Tìm hướng giống nhất -> Ghi nhận -> Trừ đi -> Lặp lại.
        """

        # 1. Chuẩn bị Target (Bias của Delta W hiện tại)
        # Lấy gradient lớp cuối cùng (bias), giữ nguyên độ lớn (magnitude)
        target_vector = approx_diff[-1].detach().cpu().numpy().flatten()
        
        # 2. Chuẩn bị Cơ sở (Basis) từ Gradient đại diện
        basis_vectors = []
        for g in representative_gradients:
            # Lấy bias của từng class
            bias_grad = normalize_to_unit(g[-1].detach().cpu().numpy().flatten())
            # [QUAN TRỌNG] Vector cơ sở PHẢI chuẩn hóa về 1
            # Để tích vô hướng (dot product) phản ánh đúng độ tương đồng (cosine)
            basis_vectors.append(normalize_to_unit(bias_grad))
            
        # Tạo ma trận Basis [10, 10] (10 chiều bias, 10 class)
        Basis = np.stack(basis_vectors, axis=1)

        # 3. THUẬT TOÁN PEELING (Bóc tách)
        
        # Khởi tạo phần dư (Residual) ban đầu chính là Target
        residual = target_vector.copy()
        
        # Mảng đếm số lượng nhãn
        counts = np.zeros(10, dtype=int)
        
        # print(f"  > Bắt đầu bóc tách Bias ({batch_size} vòng lặp)...")
        
        for step in range(batch_size):
            # a. Tính điểm tương đồng (Dot Product / Projection)
            # scores[i] = Độ lớn hình chiếu của Residual lên Class i
            scores = np.dot(Basis.T, residual) 
            
            # b. Chọn class có điểm cao nhất (Thằng trùng hướng nhất)
            best_idx = np.argmax(scores)
            
            # c. Ghi nhận kết quả
            counts[best_idx] += 1
            
            # d. Loại bỏ (Peel off)
            # Tìm vector thành phần của class đó để trừ đi
            # Công thức: v_component = (Residual . Basis_i) * Basis_i
            projection_val = scores[best_idx]
            
            # Nếu projection âm (ngược hướng hoàn toàn), có thể là nhiễu hoặc sai số, 
            # nhưng thuật toán tham lam vẫn sẽ trừ đi theo toán học.
            component_to_remove = projection_val * Basis[:, best_idx]
            print("Score: ", projection_val)
            print("Vector Class: ", Basis[:, best_idx])
            # Cập nhật phần dư cho vòng lặp sau
            residual = residual - component_to_remove
            print("residual: ", residual)

            # (Debug) Xem nó chọn gì ở từng bước
            # print(f"    Step {step+1}: Chọn Class {best_idx} (Score: {projection_val:.5f})")

        return counts
    
    def predict_label_distribution_bias_peeling_threshold(approx_diff, representative_gradients, batch_size, threshold=1e-6, max_iter=1000):
        """
        Dự đoán phân phối nhãn dùng Bias Gradient + Peeling (Dừng theo Threshold).
        
        Logic mới:
        1. Lặp liên tục để bóc tách năng lượng từ Residual.
        2. Cộng dồn giá trị chiếu (projection score) vào từng class.
        3. Dừng khi Residual quá nhỏ (dưới threshold) hoặc vượt quá số vòng lặp an toàn.
        4. Dùng hàm prepare_and_round để chia tỷ lệ và làm tròn tổng score thu được.
        """

        # 1. Chuẩn bị Target (Giữ nguyên độ lớn)
        target_vector = approx_diff[-1].detach().cpu().numpy().flatten()
    
            
        # 2. Chuẩn bị Cơ sở (Basis) - Chuẩn hóa về 1
        basis_vectors = []
        for g in representative_gradients:
            bias_grad = g[-1].detach().cpu().numpy().flatten()
            basis_vectors.append(normalize_to_unit(bias_grad))
            
        # Ma trận Basis [10, 10]
        Basis = np.stack(basis_vectors, axis=1)

        # 3. THUẬT TOÁN PEELING
        
        residual = normalize_to_unit(target_vector.copy())
        
        # [THAY ĐỔI] Mảng này giờ chứa số thực (float) để tích lũy trọng số, không phải số nguyên đếm
        accumulated_scores = np.zeros(10, dtype=float)
        
        # Tính Norm ban đầu để làm căn cứ (tùy chọn, để debug)
        initial_norm = np.linalg.norm(target_vector)
        
        # Vòng lặp
        for step in range(max_iter):
            # Kiểm tra điều kiện dừng: Nếu độ lớn phần dư nhỏ hơn ngưỡng
            current_norm = np.linalg.norm(residual)
            if current_norm < threshold:
                # print(f"  -> Converged at step {step}. Residual norm: {current_norm:.6f}")
                break
                
            # a. Tính điểm tương đồng
            scores = np.dot(Basis.T, residual)
            
            # b. Chọn class tốt nhất
            best_idx = np.argmax(scores)
            projection_val = scores[best_idx]
            
            # [QUAN TRỌNG] Nếu projection âm hoặc quá nhỏ, nghĩa là không còn bóc được gì có ý nghĩa
            # Ta cũng nên dừng để tránh cộng nhiễu hoặc trừ sai hướng
            if projection_val <= 1e-9:
                # print(f"  -> Stopped due to non-positive projection at step {step}.")
                break

            # c. [THAY ĐỔI] Cộng dồn trọng số (Score) thay vì đếm +1
            accumulated_scores[best_idx] += projection_val
            
            # d. Loại bỏ thành phần đó khỏi Residual
            component_to_remove = projection_val * Basis[:, best_idx]
            residual = residual - component_to_remove
            
        # 4. FINALIZATION: Chuyển đổi Score tích lũy thành Số lượng ảnh (Integer)
        # Hàm prepare_and_round sẽ lo việc chuẩn hóa tỷ lệ và làm tròn sao cho tổng = batch_size
        final_counts = prepare_and_round(accumulated_scores, batch_size)
        
        return final_counts
    
    def predict_label_distribution_cosine_weighted(approx_diff, representative_gradients, batch_size):
        """
        Dự đoán phân phối nhãn dựa trên TRỌNG SỐ COSINE.
        
        Logic:
        1. Tổng hợp DeltaW và Gradient mẫu thành các vector đặc trưng (1 chiều).
        2. Tính Cosine Similarity giữa DeltaW và từng Gradient mẫu.
        3. Cosine Score chính là "Trọng số thô".
        4. Loại bỏ các Score âm (ngược hướng).
        5. Chia tỷ lệ và làm tròn để tổng bằng batch_size.
        """

        # --- 1. PREPROCESSING: Aggregation (Dùng lại logic Norm -> Sum cột cho sạch nhiễu) ---
        # Bạn có thể đổi thành .flatten() nếu muốn dùng Bias thuần túy
        def aggregate_vector(grad_item):
            # Nếu là Bias (Tensor 1 chiều)
            if grad_item[-1].dim() == 1: 
                return grad_item[-1].detach().cpu().numpy().flatten()
            
            # Nếu muốn dùng Weights (Tensor 2 chiều [10, 512]) với logic Norm-Sum tối ưu
            weight_matrix = grad_item[-2].detach().cpu().numpy()
            # Chuẩn hóa từng cột feature rồi cộng lại (Logic Voting)
            norm_cols = weight_matrix / (np.linalg.norm(weight_matrix, axis=0, keepdims=True) + 1e-12)
            return np.sum(norm_cols, axis=1)

        # Lấy vector Target (Delta W)
        target_raw = aggregate_vector(approx_diff)
        # [QUAN TRỌNG] Target phải chuẩn hóa để tính Cosine
        target_vec = normalize_to_unit(target_raw)

        # Lấy vector Basis (Gradient đại diện)
        basis_vectors = []
        for g in representative_gradients:
            vec = aggregate_vector(g)
            basis_vectors.append(normalize_to_unit(vec))
        
        # Ma trận Basis [10, 10] (hoặc kích thước vector tương ứng)
        Basis = np.stack(basis_vectors, axis=1)

        # --- 2. TÍNH COSINE SIMILARITY ---
        # Công thức: Cosine = A . B (khi cả 2 đã là unit vector)
        # Kết quả: Mảng 10 phần tử, giá trị từ -1 đến 1
        cosine_scores = np.dot(Basis.T, target_vec)

        # --- 3. XỬ LÝ TRỌNG SỐ ---
        
        # a. ReLU (Rectified Linear Unit): Loại bỏ giá trị âm
        # Nếu cosine < 0 nghĩa là vector tổng đang ngược hướng với class đó -> Không thể có class đó
        raw_weights = np.maximum(cosine_scores, 0.0)
        
        # b. (Tùy chọn) Làm mềm hoặc làm cứng phân phối
        # Nếu muốn nhấn mạnh các class có cosine lớn, bạn có thể mũ phương lên
        # raw_weights = raw_weights ** 2 

        # --- 4. LÀM TRÒN (ROUNDING) ---
        # Sử dụng hàm làm tròn bảo toàn tổng (Largest Remainder Method)
        
        final_counts = prepare_and_round(raw_weights, batch_size)
        
        return final_counts
    
    def predict_label_distribution_peeling_angle_stop(approx_diff, representative_gradients, batch_size, angle_threshold=1, dominance_threshold=0.02):
        """
        Dự đoán phân phối nhãn dùng Peeling + Dừng dựa trên Góc (Angle Stop) + Hồi phục mẫu thiếu.
        
        Logic:
        1. Bóc tách từng bước.
        2. Sau khi trừ, so sánh hướng vector Residual cũ và mới.
        3. Nếu Cosine Similarity > angle_threshold (hướng không đổi) -> DỪNG.
        4. Nếu số lượng tìm được < batch_size -> Dò lại lịch sử Score để điền nốt.
        (Nguyên tắc: Chọn các lần bóc tách có sự chênh lệch (dominance) lớn nhất).
        """

        # --- 1. PREPROCESSING ---
        target_vector = approx_diff[-1].detach().cpu().numpy().flatten()
            
        basis_vectors = []
        for g in representative_gradients:
            bias_grad = g[-1].detach().cpu().numpy().flatten()
            # Basis chuẩn hóa
            basis_vectors.append(normalize_to_unit(bias_grad))
            
        Basis = np.stack(basis_vectors, axis=1)

        # --- 2. PEELING LOOP ---
        
        residual = target_vector.copy()
        counts = np.zeros(10, dtype=int)
        
        # Lưu lịch sử để dùng cho bước điền bù
        # Mỗi phần tử lưu: (step_index, best_class_idx, gap_score, full_scores)
        history_logs = []
        
        print(f"--- Bắt đầu Peeling (Max {batch_size} steps) ---")
        cos_sim = 0
        print(Basis)
        for step in range(batch_size):
            # a. Tính Scores
            scores = np.dot(Basis.T, residual)
            
            # b. Tìm Best Class
            sorted_indices = np.argsort(scores)
            best_idx = sorted_indices[-1]      # Class điểm cao nhất
            second_best_idx = sorted_indices[-2] # Class điểm nhì
            
            projection_val = scores[best_idx]
            gap = scores[best_idx] - scores[second_best_idx] # Độ chênh lệch
            
            # Lưu lại lịch sử
            history_logs.append({
                'step': step,
                'best_idx': best_idx,
                'gap': gap,
                'scores': scores
            })
            
            # c. Update Count tạm thời
            counts[best_idx] += 1
            
            # d. Tính Residual mới
            component_to_remove = projection_val * Basis[:, best_idx]
            residual_new = residual - component_to_remove
            
            # [LOGIC MỚI] e. Kiểm tra góc dừng (Angle Stop Condition)
            # Tính Cosine giữa Residual Cũ và Mới
            norm_old = np.linalg.norm(residual)
            norm_new = np.linalg.norm(residual_new)
            
            # Tránh chia cho 0
            if norm_old < 1e-9 or norm_new < 1e-9:
                print(f"  -> Dừng tại bước {step}: Residual đã về 0.")
                break
                
            if cos_sim > angle_threshold:
                print(f"  -> Dừng sớm tại bước {step}: Hướng thay đổi không đáng kể (Cos={cos_sim:.6f}).")
                # [QUAN TRỌNG] Hoàn tác bước cộng count vừa rồi vì bước này được coi là vô nghĩa
                counts[best_idx] -= 1 
                break
            
            
            dot_prod = np.dot(residual, residual_new)
            cos_sim = dot_prod / (norm_old * norm_new)
            
            print("Cos_sim: ",cos_sim /np.linalg.norm(residual_new) )
            print(residual_new)
            print(scores)
            print(Basis[:, best_idx])

            # Clip giá trị cos trong [-1, 1] để tránh lỗi số học
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            
            # print(f"  Step {step}: Class {best_idx} | Score {projection_val:.4f} | Cos Change: {cos_sim:.6f}")
            
            # Nếu cos quá gần 1 (góc ~ 0 độ) -> Việc trừ không thay đổi hướng -> Dừng
            
                
            # Cập nhật residual thật
            residual = residual_new

        # --- 3. RECOVERY PHASE (Điền bù mẫu thiếu) ---
        
        current_total = np.sum(counts)
        missing = batch_size - current_total
        
        if missing > 0:
            print(f"  -> Còn thiếu {missing} mẫu. Tiến hành dò lại lịch sử...")
            
            # Duyệt lại lịch sử từ đầu (step 0, 1, 2...)
            for log in history_logs:
                if missing == 0:
                    break
                    
                idx = log['best_idx']
                gap = log['gap']
                
                # Kiểm tra độ chênh lệch (Dominance)
                # Nếu sự chênh lệch giữa Top 1 và Top 2 đủ lớn -> Đây là một dự đoán chắc chắn
                # Ta có thể tự tin gán thêm sample vào class này
                if gap > dominance_threshold:
                    print(f"    + Bù 1 mẫu vào Class {idx} (Do Gap lớn: {gap:.4f})")
                    counts[idx] += 1
                    missing -= 1
                else:
                    pass 
                    # print(f"    - Bỏ qua step {log['step']} (Gap quá nhỏ: {gap:.4f})")

            # Nếu chạy hết lịch sử mà vẫn còn thiếu (trường hợp hiếm)
            # Gán nốt vào class có tổng điểm tích lũy cao nhất (Fallback)
            if missing > 0:
                print(f"  -> Vẫn thiếu {missing} mẫu. Gán vào class có điểm tích lũy cao nhất.")
                # Tính tổng score của các step đã qua
                total_scores = np.zeros(10)
                for log in history_logs:
                    total_scores += log['scores']
                
                top_class = np.argmax(total_scores)
                counts[top_class] += missing

        return counts
    
    # --------------------------------------------------------------------------
    # PHẦN MỚI: TÍNH GRADIENT ĐẠI DIỆN TỪ PROBING SAMPLES (CIFAR-10)
    # --------------------------------------------------------------------------
    print("\n[INFO] Bắt đầu tính Gradient đại diện cho từng lớp từ Probing Samples...")
    


    # 1. Cấu hình đường dẫn và Mapping
    # Lưu ý: Thay đổi đường dẫn này trỏ đến đúng thư mục cha chứa các folder bird, car...
    PROBING_ROOT_DIR = '/kaggle/input/probing-sampels-cifar10/probing_results' 
    
    # Mapping chuẩn CIFAR-10: Index 0->9
    # Tên trong list phải KHỚP CHÍNH XÁC với tên folder của bạn
    cifar10_folder_mapping = [
        'plane', 'car', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Vector lưu trữ Gradient đại diện (List of Gradients)
    # class_representative_gradients[i] sẽ chứa Gradient của class i
    class_representative_gradients = []

    # Hàm transform cho ảnh đầu vào (Dùng lại dmlist, dslist của file gốc)
    # Lưu ý: Probing samples là ảnh PNG thường (0-255), cần ToTensor() trước khi Normalize
    probing_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Ensure size matches model input (32x32)
        transforms.ToTensor(),
        transforms.Normalize(dmlist, dslist)
    ])

    # Model để tính toán (Sử dụng model_ft - model đã fine-tune)
    model_ft.eval()
    model_ft.to(**setup)

    # 2. Vòng lặp qua 10 lớp theo đúng thứ tự 0 -> 9
    for class_idx, folder_name in enumerate(cifar10_folder_mapping):
        folder_path = os.path.join(PROBING_ROOT_DIR, folder_name)
        
        # Lấy danh sách tất cả file png trong folder
        image_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        
        if len(image_files) == 0:
            print(f"Warning: Không tìm thấy ảnh trong folder {folder_name}")
            continue

        print(f"Processing Class {class_idx} ({folder_name}): {len(image_files)} images...")

        # Load và Preprocess ảnh -> Gom thành 1 Batch
        batch_images = []
        for img_file in image_files:
            img = Image.open(img_file).convert('RGB')
            img_tensor = probing_transform(img)
            batch_images.append(img_tensor)
        
        # Stack thành Tensor: (Batch_Size, 3, 32, 32)
        inputs = torch.stack(batch_images).to(**setup)
        
        # Tạo nhãn (Labels): Tất cả ảnh trong folder này đều có nhãn là class_idx
        labels = torch.full((len(batch_images),), class_idx, dtype=torch.long).to(setup['device'])

        # 3. Tính Gradient đại diện bằng hàm Gradient_Cal_only
        # Lưu ý: Hàm Gradient_Cal_only bạn đã thêm vào recovery_algo.py
        # Kết quả trả về là Gradient trung bình của cả 10 ảnh (do hàm loss mặc định là mean)
        grads = rs.recovery_algo.Gradient_Cal_only(model_ft, inputs, labels)
        
        # Detach và chuyển về CPU để tiết kiệm VRAM nếu cần lưu trữ lâu dài
        grads_cpu = tuple(g.detach().cpu() for g in grads)
        
        class_representative_gradients.append(grads_cpu)
    print(f"[SUCCESS] Đã tính xong Gradient đại diện cho {len(class_representative_gradients)} lớp.")
    my_custom_vector_plane =    torch.tensor([-1.0 , 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    my_custom_vector_car =      torch.tensor([ 0.05, -1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    my_custom_vector_bird =     torch.tensor([ 0.05, 0.05, -1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    my_custom_vector_cat =      torch.tensor([ 0.05, 0.05,  0.05, -1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    my_custom_vector_deer =     torch.tensor([ 0.05, 0.05,  0.05, 0.05, -1.0, 0.05, 0.05, 0.05, 0.05, 0.05])
    my_custom_vector_dog =      torch.tensor([ 0.05, 0.05,  0.05, 0.05, 0.05,-1.0, 0.05, 0.05, 0.05, 0.05])
    my_custom_vector_frog =     torch.tensor([ 0.05, 0.05,  0.05, 0.05, 0.05, 0.05,-1.0, 0.05, 0.05, 0.05])
    my_custom_vector_horse=     torch.tensor([ 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,-1.0,  0.05, 0.05])
    my_custom_vector_ship =     torch.tensor([ 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, -1.0, 0.05])
    my_custom_vector_truck =    torch.tensor([ 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, -1.0,])
    class_representative_gradients[0][-1].data = my_custom_vector_plane
    class_representative_gradients[1][-1].data = my_custom_vector_car
    class_representative_gradients[2][-1].data = my_custom_vector_bird
    class_representative_gradients[3][-1].data = my_custom_vector_cat
    class_representative_gradients[4][-1].data = my_custom_vector_deer
    class_representative_gradients[5][-1].data = my_custom_vector_dog
    class_representative_gradients[6][-1].data = my_custom_vector_frog
    class_representative_gradients[7][-1].data = my_custom_vector_horse
    class_representative_gradients[8][-1].data = my_custom_vector_ship
    class_representative_gradients[9][-1].data = my_custom_vector_truck
    # # Lưu lại kết quả nếu cần
    torch.save(class_representative_gradients, os.path.join(save_folder, 'class_rep_gradients.pth'))


    print("Exact unlearn each sample and test the exact and approximate unlearn")
    # --------------------------------------------------------------------------------
    #------------------------Kết thúc-----------------------------------------
    #-------------------------------------------------------------------------
    model_ft.zero_grad()
    model_ft.to(**setup)
    rec_machine_ft = rs.GradientReconstructor(model_ft, (dm, ds), recons_config, num_images=args.unlearn_samples)
    
    model_pretrain.zero_grad()
    model_pretrain.to(**setup)
    rec_machine_pretrain = rs.GradientReconstructor(model_pretrain, (dm, ds), recons_config, num_images=args.unlearn_samples)
    np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.5e}\t"})

    total_acc = 0.0  
    # for test_id in range(args.ft_samples // args.unlearn_samples):
    total_loop = 150
    for test_id in range(20):
        unlearn_ids = list(range(test_id * args.unlearn_samples, (test_id + 1) * args.unlearn_samples))
        print(f"Unlearn {unlearn_ids}")
        unlearn_folder = os.path.join(save_folder, f'unlearn_ft_batch{test_id}')
        os.makedirs(unlearn_folder, exist_ok=True)
        X_list = [xt for i, xt in enumerate(X_all) if i not in unlearn_ids]
        if len(X_list) > 0:
            X = torch.stack([xt for i, xt in enumerate(X_all) if i not in unlearn_ids])
            y = torch.tensor([yt for i, yt in enumerate(y_all) if i not in unlearn_ids])
            print("Exact unlearn data size", X.shape, y.shape)
            trainset_unlearn = rs.data_processing.SubTrainDataset(X, y, transform=transforms.Normalize(dmlist, dslist))
            trainloader_unlearn = torch.utils.data.DataLoader(trainset_unlearn, batch_size=min(defs.batch_size, len(trainset_unlearn)), shuffle=True, num_workers=8, pin_memory=True)
        
        X_unlearn = torch.stack([xt for i, xt in enumerate(X_all) if i in unlearn_ids])
        y_unlearn = torch.tensor([yt for i, yt in enumerate(y_all) if i in unlearn_ids])

        print(f"***** Train unlearned model (withouth {unlearn_ids}) *****")
        # model_unlearn, _ = rs.construct_model(args.model, num_classes=num_classes, num_channels=3)
        # model_unlearn.load_state_dict(state_dict)
        # model_unlearn.eval()
        # model_unlearn.to(**setup)
        # if len(X_list) > 0:
        #     unlearn_stats = rs.train(model_unlearn, loss_fn, trainloader_unlearn, validloader, defs, setup=setup, ckpt_path=unlearn_folder, finetune=True)
        # else:
        #     unlearn_stats = None
        # model_unlearn.cpu()
        # resdict = {'tr_args': args.__dict__,
        #     'tr_strat': defs.__dict__,
        #     'stats': unlearn_stats,
        #     'unlearn_batch_id': test_id}
        # torch.save(resdict, os.path.join(unlearn_folder, 'finetune_params.pth'))
        # # unlearn_params =  [param.detach() for param in model_unlearn.parameters()]
        # un_diffs = [(un_param.detach().cpu() - org_param.detach().cpu()).detach() for (un_param, org_param) in zip(model_unlearn.parameters(), model_pretrain.parameters())]

        # print("Start reconstruction.")
        


        recons_folder = os.path.join(save_folder, 'recons')
        figure_folder = os.path.join(save_folder, 'figures')
        os.makedirs(recons_folder , exist_ok=True)
        os.makedirs(figure_folder, exist_ok=True)
        # reconstruction
        
        
        # exact_diff = [-(ft_diff * args.ft_samples - un_diff * len(X_list)).detach().to(**setup) for (ft_diff, un_diff) in zip(ft_diffs, un_diffs)]
        # rec_machine_pretrain.model.eval()
        # result_exact = rec_machine_pretrain.reconstruct(exact_diff, normalizer(X_unlearn.to(**setup)), y_unlearn.to(setup['device']), img_shape=(3, img_size, img_size))
        # process_recons_results(result_exact, X_unlearn, figpath=figure_folder, recons_path=recons_folder, filename=f'exact{test_id}_{index[test_id].item()}')

        approx_diff = [p.detach().to(**setup) for p in rs.recovery_algo.loss_steps(model_ft, normalizer(X_unlearn.to(**setup)), y_unlearn.to(setup['device']), lr=1, local_steps=1)] # lr is not important in cosine 
        weights_after =  [p.detach().to(**setup) for p in rs.recovery_algo.weights_after(model_ft, normalizer(X_unlearn.to(**setup)), y_unlearn.to(setup['device']), lr=1, local_steps=1)] # lr is not important in cosine 
        delta_W  = approx_diff[-2]
        new_W = weights_after[-2]
        dot_product_vector = torch.sum(delta_W * new_W, dim=1)
        result_10_elements = dot_product_vector.detach().cpu().numpy()
        print("Vector kết quả (10 phần tử):")
        print(result_10_elements)
        print(f"Shape gốc: {delta_W.shape}") # Nên là torch.Size([10, 512])
        print(f"Shape kết quả: {result_10_elements.shape}") # Nên là (10,)
        sum_delta = 0
        all_deltas = rs.recovery_algo.loss_steps_each_corrected(model_ft, normalizer(X_unlearn.to(**setup)), y_unlearn.to(setup['device']), lr=1)
        
        for delta in all_deltas:
            vector =   delta[-1].detach().cpu().numpy().flatten() 
            sum_delta += vector# Cộng tensor bias của từng ảnh
            print(vector)
        mean_delta = sum_delta / len(all_deltas)
        # print("Delta Batch Gốc:     ", normalize_to_unit(approx_diff[-1].detach().cpu().numpy().flatten()))
        print("Mean Delta Từng Ảnh: ", mean_delta)
        # Chia cho số lượng ảnh
        # So sánh       
        
        if class_representative_gradients is not None:
            # print("\n--- Label Recovery Result ---")
            
            # 1. Thực hiện dự đoán
    
            predicted_counts = predict_label_distribution_bias_peeling(approx_diff, class_representative_gradients, args.unlearn_samples)
            
            # 2. Chuyển đổi Ground Truth (y_unlearn) sang dạng đếm (Counts) để dễ so sánh
            actual_counts = np.zeros(10, dtype=int)
            y_unlearn_cpu = y_unlearn.cpu().numpy()
            for y in y_unlearn_cpu:
                actual_counts[y] += 1
            
            # 3. In kết quả so sánh
            # Mapping tên class (nếu muốn đẹp)
            classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            
            print(f"Ground Truth Labels (Raw): {y_unlearn_cpu}") # <--- YÊU CẦU CỦA BẠN
            
            print(f"{'Class':<10} | {'Real':<5} | {'Pred':<5} | {'Diff'}")
            print("-" * 35)
            correct_count = 0
            for i in range(10):
                diff = predicted_counts[i] - actual_counts[i]
                if actual_counts[i] > 0 or predicted_counts[i] > 0: # Chỉ in những class có xuất hiện
                    print(f"{classes[i]:<10} | {actual_counts[i]:<5} | {predicted_counts[i]:<5} | {diff}")
                
                # Tính độ chính xác đơn giản (Total Variation Distance / 2)
                correct_count += min(actual_counts[i], predicted_counts[i])
            
            acc = correct_count / args.unlearn_samples * 100
            total_acc += acc
            print(f"--> Batch Accuracy: {acc:.2f}%")
            # print("-" * 35)
        
       
        # rec_machine_ft.model.eval()
        # result_approx = rec_machine_ft.reconstruct(approx_diff, normalizer(X_unlearn.to(**setup)), y_unlearn.to(setup['device']), img_shape=(3, img_size, img_size))
        # process_recons_results(result_approx, X_unlearn, figpath=figure_folder, recons_path=recons_folder, filename=f'approx{test_id}_{index[test_id].item()}')
    print(f"--> Total Accuracy: {(total_acc/total_loop):.2f}%") 
        




        