#!/usr/bin/env python
# coding: utf-8

"""
Vision Transformer (ViT) 模型训练和测试脚本

基于CNN.py的架构，使用Vision Transformer替代CNN模型
保持相同的输入输出格式、训练流程和测试方式
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")


# ============================================================================
# 模型定义
# ============================================================================

class PatchEmbed3D(nn.Module):
    """3D图像转换为patch embeddings"""
    def __init__(self, input_dim, embed_dim, patch_size=(2, 3, 3)):
        super().__init__()
        self.patch_size = patch_size
        # 使用stride=patch_size进行下采样
        self.proj = nn.Conv3d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, D, H, W) = (B, 6, 14, 45, 60)
        # 14/2=7, 45/3=15, 60/3=20
        x = self.proj(x)  # (B, embed_dim, 7, 15, 20)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 7*15*20, embed_dim) = (B, 2100, embed_dim)
        return x, (D, H, W)


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3D(nn.Module):
    """3D Vision Transformer模型"""
    def __init__(self, input_dim, output_dim, embed_dim=256, depth=6, num_heads=8, 
                 patch_size=(2, 3, 3), mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(input_dim, embed_dim, patch_size)
        
        # Positional embedding (learnable)
        # 输入是 (B, 6, 14, 45, 60)，patch_size=(2,3,3)后是 (7, 15, 20) -> 2100个patches
        # 14/2=7, 45/3=15, 60/3=20
        self.pos_embed = nn.Parameter(torch.zeros(1, 2100, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder: 将patch embeddings还原为3D输出
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, output_dim * patch_size[0] * patch_size[1] * patch_size[2])
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # x: (B, input_dim, D, H, W) = (B, 6, 12, 46, 91)
        B = x.shape[0]
        
        # Patch embedding
        x, (D, H, W) = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Decode
        x = self.decoder(x)  # (B, num_patches, output_dim * patch_volume)
        
        # Reshape to 3D
        x = x.reshape(B, D, H, W, self.output_dim, 
                     self.patch_size[0], self.patch_size[1], self.patch_size[2])
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # (B, output_dim, D, patch_d, H, patch_h, W, patch_w)
        x = x.reshape(B, self.output_dim, 
                     D * self.patch_size[0], 
                     H * self.patch_size[1], 
                     W * self.patch_size[2])
        
        return x


# ============================================================================
# 工具类（与CNN.py相同）
# ============================================================================

class random_seed():
    
    def __init__(self,seed = 2023, benchmark = False, deterministic = False):
        
        self.seed = seed
        self.benchmark = benchmark
        self.deterministic = deterministic
        
    def set_seed(self):
        
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            
        np.random.seed(self.seed)
        
        torch.backends.cudnn.benchmark = self.benchmark
        torch.backends.cudnn.deterministic = self.deterministic


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_dict, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(
            model_dict, save_path+'/checkpoint.pth')
        self.val_loss_min = val_loss


# ============================================================================
# 数据处理类（与CNN.py相同）
# ============================================================================

class CustomDataset:
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        if self.transform:
            data = self.transform(data)
        return data.astype(np.float32)


class Load_Data():
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def get_list_by_indices(self, file_path, start_idx, end_idx):
        """根据索引范围获取文件列表"""
        files = sorted(os.listdir(file_path), 
                      key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        selected_files = []
        for file in files:
            file_num = int(file.split('_')[1].split('.')[0])
            if start_idx <= file_num < end_idx:
                selected_files.append(file)
        
        data_list = [os.path.join(file_path, f) for f in selected_files]
        return data_list
    
    def get_list_by_ratio(self, file_path, start_seg, end_seg):
        """根据比例获取文件列表"""
        files = sorted(os.listdir(file_path), 
                      key=lambda x: int(x.split('_')[1].split('.')[0]))
        files = files[int(len(files)*start_seg):int(len(files)*end_seg)]
        data_list = [os.path.join(file_path, f) for f in files]
        return data_list
    
    def load_data_by_indices(self, start_idx, end_idx, custom_transform=None):
        """根据索引范围加载数据"""
        data_y_list = self.get_list_by_indices(self.output_path, start_idx, end_idx)
        print(f"找到 {len(data_y_list)} 个文件在范围 [{start_idx}, {end_idx})")
        
        data_y = CustomDataset(data_y_list, transform=custom_transform)
        
        all_data = np.stack([data_y[i] for i in range(len(data_y))], axis=0).astype(np.float32)
        return all_data
    
    def load_data_by_ratio(self, start_seg, end_seg, custom_transform=None):
        """根据比例加载数据"""
        data_y_list = self.get_list_by_ratio(self.output_path, start_seg, end_seg)
        data_y = CustomDataset(data_y_list, transform=custom_transform)
        
        all_data = np.stack([data_y[i] for i in range(len(data_y))], axis=0)
        return all_data


# ============================================================================
# 训练相关类（与CNN.py相同）
# ============================================================================

class Train():
    
    def __init__(self, net, epochs, lr, batch_size, filename, train_x, train_y, val_x, val_y, save_dir, patience, gpu_ids, num_workers = 8, test_x=None, test_y=None, accumulation_steps=1):
        
        self.net = net
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.filename = filename
        self.patience = patience
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y
        self.save_dir = save_dir
        self.gpu_ids = gpu_ids
        self.num_workers = num_workers
        self.accumulation_steps = accumulation_steps  # 梯度累积步数
        
        
    def print_use_time(self,time_use):
        str_leng = '#'+"Model training time spent {:.0f}h {:.0f}m".format(time_use // 3600, (time_use % 3600) // 60)+'#'
        count = 0
        while str_leng[count:]:
            count += 1
        print('\n')
        for i in range(3):
            if i == 1:
                print('#',"Model training time spent {:.0f}h {:.0f}m".format(time_use // 3600, (time_use % 3600) // 60),'#')
            else:
                print('=' * (count+2))

    def train(self):

        run_dir = os.path.join(self.save_dir, 'runs')
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        tb = SummaryWriter(run_dir)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        if len(self.gpu_ids) > 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(device)
            self.net = nn.DataParallel(self.net, device_ids=list(range(len(self.gpu_ids))))
        else:
            device = torch.device(f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
            self.net.to(device)

        lossfunction = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,  weight_decay=1e-5)
        
        if os.path.exists(os.path.join(self.save_dir, 'checkpoint.pth')):
            print('==> loading existing model')
            model_info = torch.load(os.path.join(self.save_dir, 'checkpoint.pth'))
            self.net.load_state_dict(model_info['state_dict'])
            optimizer.load_state_dict(model_info['optimizer'])
            cur_epoch = model_info['epoch'] + 1
        else:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            cur_epoch = 0
        pla_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          factor=0.5,
                                                          patience=4,
                                                          verbose=True)


        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = []

        since = time.time()

        for epoch in range(cur_epoch, self.epochs + 1):

            train_X = torch.utils.data.DataLoader(self.train_x, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            train_Y = torch.utils.data.DataLoader(self.train_y, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

            t = tqdm(zip(train_X,train_Y), leave=False, total=len(train_X))
            
            for idx, (inputVar, targetVar) in enumerate(t):

                inputs = inputVar.to(device)
                label = targetVar.to(device)
                
                self.net.train()
                pred = self.net(inputs)
                loss = lossfunction(pred, label)
                loss = loss / self.accumulation_steps  # 归一化损失
                loss_aver = loss.item() * self.accumulation_steps / self.batch_size
                train_losses.append(loss_aver)
                loss.backward()
                
                # 梯度累积：每 accumulation_steps 步更新一次
                if (idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_value=10.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                t.set_postfix({
                    'trainloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })
            
            tb.add_scalar('TrainLoss', loss_aver, epoch)
            
            with torch.no_grad():
                self.net.eval()

                val_X = torch.utils.data.DataLoader(self.val_x, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                val_Y = torch.utils.data.DataLoader(self.val_y, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

                t = tqdm(zip(val_X,val_Y), leave=False, total=len(val_X),colour = 'pink')
                for inputVar,targetVar in t:

                    inputs = inputVar.to(device)
                    label = targetVar.to(device)
                    pred = self.net(inputs)
                    loss = lossfunction(pred, label)
                    loss_aver = loss.item() / self.batch_size
                    valid_losses.append(loss_aver)
                    t.set_postfix({
                        'validloss': '{:.6f}'.format(loss_aver),
                        'epoch': '{:02d}'.format(epoch)
                    })

            tb.add_scalar('ValidLoss', loss_aver, epoch)
            
            torch.cuda.empty_cache()
            
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(self.epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.6f} ' +
                         f'valid_loss: {valid_loss:.6f}')

            print(print_msg)
            train_losses = []
            valid_losses = []
            pla_lr_scheduler.step(valid_loss)
            model_dict = {
                'epoch': epoch,
                'state_dict': self.net.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            early_stopping(valid_loss, model_dict, epoch, self.save_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.print_use_time(time.time() - since)

        loss_data = {
            'avg_train_loss':avg_train_losses,
            'avg_valid_loss':avg_valid_losses
        }
        df_loss = pd.DataFrame(loss_data)

        loss_dir = os.path.join(self.save_dir, 'loss')
        
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
            
        df_loss.to_csv(os.path.join(loss_dir, self.filename+'.csv'))


class Show_Loss():
    
    def __init__(self, filename, save_dir):
        
        self.filename = filename
        self.save_dir = save_dir
        
    def loss_image_save(self):
        
        loss_dir = os.path.join(self.save_dir, 'loss')
        load_path = os.path.join(loss_dir, self.filename+'.csv')
        save_path = os.path.join(loss_dir, self.filename+'.jpg')
        
        df_loss = pd.read_csv(load_path, index_col=0)

        fig = plt.figure(figsize = (9,5),dpi = 80)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(df_loss.avg_train_loss, label='train_loss')
        ax.plot(df_loss.avg_valid_loss, label='valid_loss')

        ax.grid(True, linestyle='--', alpha=0.7)

        ax.set_title('Training and Validation Loss Over Time')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        
        plt.savefig(save_path,bbox_inches = 'tight',dpi=100)


# ============================================================================
# 测试相关类（与CNN.py相同）
# ============================================================================


class Test():
    """模型测试类（归一化数据）"""
    
    def __init__(self, net, batch_size, filename, test_x, test_y, save_dir, gpu_ids, num_workers=8):
        self.net = net
        self.batch_size = batch_size
        self.filename = filename
        self.test_x = test_x
        self.test_y = test_y
        self.save_dir = save_dir
        self.gpu_ids = gpu_ids
        self.num_workers = num_workers
        
    def test(self):
        """在测试集上评估模型（归一化数据）"""
        
        if len(self.gpu_ids) > 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {device}")
        
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        if any(k.startswith('module.') for k in state_dict.keys()):
            print("Removing 'module.' prefix from checkpoint...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.net.to(device)
        self.net.load_state_dict(state_dict, strict=False)
        
        if len(self.gpu_ids) > 1 and torch.cuda.is_available():
            self.net = nn.DataParallel(self.net, device_ids=list(range(len(self.gpu_ids))))
        
        self.net.eval()
        
        test_X = torch.utils.data.DataLoader(self.test_x, batch_size=self.batch_size, 
                                            shuffle=False, num_workers=self.num_workers)
        test_Y = torch.utils.data.DataLoader(self.test_y, batch_size=self.batch_size, 
                                            shuffle=False, num_workers=self.num_workers)
        
        lossfunction = nn.MSELoss().to(device)
        test_losses = []
        pre_list = []
        label_list = []
        
        print("Testing...")
        with torch.no_grad():
            t = tqdm(zip(test_X, test_Y), leave=False, total=len(test_X))
            for inputVar, targetVar in t:
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                
                pred = self.net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / self.batch_size
                test_losses.append(loss_aver)
                
                pre_list.append(pred.cpu().numpy())
                label_list.append(targetVar.numpy())
                
                t.set_postfix({'test_loss': '{:.6f}'.format(loss_aver)})
        
        predictions = np.concatenate(pre_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        
        result_dir = os.path.join(self.save_dir, 'test_results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        np.save(os.path.join(result_dir, 'predictions.npy'), predictions)
        np.save(os.path.join(result_dir, 'labels.npy'), labels)
        
        avg_test_loss = np.mean(test_losses)
        loss_df = pd.DataFrame({'test_loss': test_losses})
        loss_df.to_csv(os.path.join(result_dir, 'test_loss.csv'), index=False)
        
        print(f'\n==> Test completed!')
        print(f'Average test loss: {avg_test_loss:.6f}')
        print(f'Results saved to: {result_dir}')
        
        r2 = self._calculate_r2(labels, predictions)
        rmse = self._calculate_rmse(labels, predictions)
        mae = self._calculate_mae(labels, predictions)
        r = self._calculate_correlation(labels, predictions)
        
        print(f'\n==> Test Metrics (Normalized Data):')
        print(f'Correlation: {r:.4f}')
        print(f'R2 Score: {r2:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        
        metrics = {
            'Correlation': r,
            'R2_Score': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Avg_Loss': avg_test_loss
        }
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(os.path.join(result_dir, 'test_metrics.csv'), index=False)
        
        return predictions, labels, metrics
    
    def _calculate_r2(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        var = np.var(y_true)
        return 1 - mse / var if var != 0 else 0
    
    def _calculate_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _calculate_mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def _calculate_correlation(self, y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return np.corrcoef(y_true_flat, y_pred_flat)[0, 1]


class Model_Result():
    """模型结果处理类（反归一化+掩码）"""
    
    def __init__(self, net, save_dir, batch_size, test_x, test_y, output_mask_path, 
                 pred_path, test_y_path, gpu_ids, num_workers=8):
        self.net = net
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.test_x = test_x
        self.test_y = test_y
        self.output_mask_path = output_mask_path
        self.pred_path = pred_path
        self.test_y_path = test_y_path
        self.gpu_ids = gpu_ids
        self.num_workers = num_workers

    def output_inver(self, data):
        """反归一化处理"""
        print(f"Denormalizing data with shape: {data.shape}")
        
        min_lst = [0.0579, 0.0571, 0.0548, 0.0513, 0.0474, 0.046, 0.0504, 0.0617, 
                   0.068, 0.0587, 0.0472, 0.0381, 0.0316, 0.0261, 0.0214, 0.0182, 
                   0.0137, 0.011, 0.0075, 0.0051]
        dif_lst = [0.7557, 0.7907, 0.7054, 0.6156, 0.5476, 0.4894, 0.4408, 0.3896, 
                   0.3514, 0.3359, 0.3265, 0.3162, 0.2894, 0.269, 0.2483, 0.2166, 
                   0.1421, 0.0878, 0.0332, 0.0161]
        
        min_arr = np.array(min_lst).reshape(1, -1, 1, 1, 1)
        dif_arr = np.array(dif_lst).reshape(1, -1, 1, 1, 1)
        
        return data * dif_arr + min_arr

    def set_mask(self, data, mask_file):
        """应用掩码"""
        data_length = data.shape[0]
        mask = np.tile(mask_file, (data_length, 1, 1, 1, 1))
        
        if np.isnan(mask).any():
            mask = np.isnan(mask)
        else:
            mask = mask.astype(bool)
        
        data[mask] = np.nan
        return data
        
    def result(self):
        """执行完整的测试流程：推理 -> 反归一化 -> 掩码 -> 保存"""
        
        if len(self.gpu_ids) > 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {device}, CUDA count: {torch.cuda.device_count()}")
        
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        epoch = checkpoint.get('epoch', None)
        
        print(f"Loaded checkpoint from epoch: {epoch}")
        
        if any(k.startswith('module.') for k in state_dict.keys()):
            print("Removing 'module.' prefix from checkpoint...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.net.to(device)
        self.net.load_state_dict(state_dict, strict=False)
        
        if len(self.gpu_ids) > 1 and torch.cuda.is_available():
            print(f"Using {len(self.gpu_ids)} GPUs with DataParallel...")
            self.net = nn.DataParallel(self.net, device_ids=list(range(len(self.gpu_ids))))
        
        self.net.eval()
        
        test_X = torch.utils.data.DataLoader(self.test_x, batch_size=self.batch_size, 
                                            shuffle=False, num_workers=self.num_workers)
        test_Y = torch.utils.data.DataLoader(self.test_y, batch_size=self.batch_size, 
                                            shuffle=False, num_workers=self.num_workers)
        
        print("Running inference...")
        pre_list, label_list = [], []
        with torch.no_grad():
            t = tqdm(zip(test_X, test_Y), leave=False, total=len(test_X))
            for inputVar, targetVar in t:
                inputs = inputVar.to(device)
                predict = self.net(inputs)
                pre_list.append(predict.cpu().numpy())
                label_list.append(targetVar.numpy())
        
        pre = np.concatenate(pre_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        
        print(f"Inference completed. Shape: {pre.shape}")
        
        print("Applying denormalization...")
        pred = self.output_inver(pre)
        test_y = self.output_inver(label)
        
        if self.output_mask_path and os.path.exists(self.output_mask_path):
            print(f"Applying mask from: {self.output_mask_path}")
            output_mask = np.load(self.output_mask_path)
            pred = self.set_mask(pred, output_mask)
            test_y = self.set_mask(test_y, output_mask)
        else:
            print("Warning: No mask file provided or file not found. Skipping mask application.")
        
        print(f"Saving results...")
        np.save(self.pred_path, pred)
        np.save(self.test_y_path, test_y)
        
        print(f"✅ Results saved:")
        print(f"  - Predictions: {self.pred_path}")
        print(f"  - Labels: {self.test_y_path}")


class Metrics():
    """评估指标计算类"""
    
    def __init__(self, test_y_path, pred_path):
        self.test_y_path = test_y_path
        self.pred_path = pred_path
        
    def load_result(self):
        """加载预测结果和真实标签"""
        if not os.path.exists(self.test_y_path):
            raise FileNotFoundError(f"Label file not found: {self.test_y_path}")
        if not os.path.exists(self.pred_path):
            raise FileNotFoundError(f"Prediction file not found: {self.pred_path}")
        
        test_y = np.load(self.test_y_path)
        pred = np.load(self.pred_path)
        
        print(f"Loaded data shapes - Labels: {test_y.shape}, Predictions: {pred.shape}")
        return test_y, pred
    
    def R2_score(self, test_y, pred):
        """计算R²决定系数（支持NaN值）"""
        MSE = np.nanmean(np.power((test_y - pred), 2))
        var = np.nanvar(test_y)
        if var == 0:
            return 0
        return 1 - MSE / var

    def RMSE_score(self, test_y, pred):
        """计算均方根误差（支持NaN值）"""
        return np.sqrt(np.nanmean(np.power((test_y - pred), 2)))

    def correlation_coef(self, test_y, pred):
        """计算皮尔逊相关系数（支持NaN值）"""
        test_y_flat = test_y.flatten()
        pred_flat = pred.flatten()
        
        nan_mask = np.isnan(test_y_flat) | np.isnan(pred_flat)
        test_y_clean = test_y_flat[~nan_mask]
        pred_clean = pred_flat[~nan_mask]
        
        if len(test_y_clean) == 0:
            print("Warning: No valid data points for correlation calculation")
            return np.nan
        
        correlation = np.corrcoef(test_y_clean, pred_clean)[0, 1]
        return correlation

    def MAE(self, test_y, pred):
        """计算平均绝对误差（支持NaN值）"""
        assert test_y.shape == pred.shape, f"Shape mismatch: {test_y.shape} vs {pred.shape}"
        mae = np.nanmean(np.abs(test_y - pred))
        return mae
    
    def calculate_all_metrics(self):
        """计算所有评估指标"""
        test_y, pred = self.load_result()
        
        print("\n" + "="*50)
        print("Calculating Metrics...")
        print("="*50)
        
        r = self.correlation_coef(test_y, pred)
        r2 = self.R2_score(test_y, pred)
        rmse = self.RMSE_score(test_y, pred)
        mae = self.MAE(test_y, pred)
        
        valid_points = np.sum(~np.isnan(test_y.flatten()))
        total_points = test_y.size
        
        print(f"\nValid data points: {valid_points}/{total_points} ({valid_points/total_points*100:.2f}%)")
        print(f"\nMetrics:")
        print(f"  Correlation (r):  {r:.4f}")
        print(f"  R² Score:         {r2:.4f}")
        print(f"  RMSE:             {rmse:.4f}")
        print(f"  MAE:              {mae:.4f}")
        print("="*50 + "\n")
        
        return r, r2, rmse, mae
    
    def main(self):
        """主函数（保持向后兼容）"""
        return self.calculate_all_metrics()


# ============================================================================
# 主函数
# ============================================================================


def main(mode='train_test', gpu_ids=[0], batch_size=32, epochs=300, lr=0.0001, patience=8, 
         use_denormalize=True, output_mask_path=os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'output_mask.npy',
         embed_dim=256, depth=6, num_heads=8, accumulation_steps=1):
    """
    主函数
    mode: 'train' - 仅训练
          'test' - 仅测试
          'train_test' - 训练后测试（默认）
    gpu_ids: GPU设备ID列表，例如 [0] 或 [0,1,2]
    batch_size: 批次大小（ViT建议使用较小的batch_size）
    epochs: 训练轮数
    lr: 学习率（ViT通常需要较小的学习率）
    patience: 早停耐心值
    use_denormalize: 是否使用反归一化和掩码
    output_mask_path: 掩码文件路径
    embed_dim: Transformer嵌入维度
    depth: Transformer层数
    num_heads: 注意力头数
    accumulation_steps: 梯度累积步数（用于模拟更大的batch size）
    """
    
    input_path = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'pre_train_data/input/'
    output_path = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'pre_train_data/output/'

    filename = 'ViT2'
    save_dir = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + '' + filename

    print(f'\n==> Configuration:')
    print(f'Mode: {mode}')
    print(f'GPU IDs: {gpu_ids}')
    print(f'Batch size: {batch_size}')
    print(f'Accumulation steps: {accumulation_steps}')
    print(f'Effective batch size: {batch_size * accumulation_steps}')
    print(f'Epochs: {epochs}')
    print(f'Learning rate: {lr}')
    print(f'Patience: {patience}')
    print(f'Use denormalize: {use_denormalize}')
    print(f'Embed dim: {embed_dim}')
    print(f'Depth: {depth}')
    print(f'Num heads: {num_heads}')

    set_seeds = random_seed(seed=2023, benchmark=False, deterministic=False)
    set_seeds.set_seed()

    train_start = 1827
    train_end = 7671
    train_total = train_end - train_start
    
    train_split = train_start + int(train_total * 0.9)
    
    val_start = train_split
    val_end = train_end
    
    test_start = 7671
    test_end = 9119

    print(f'\n==> Data split:')
    print(f'Training set: [{train_start}, {train_split}) - {train_split - train_start} samples')
    print(f'Validation set: [{val_start}, {val_end}) - {val_end - val_start} samples')
    print(f'Test set: [{test_start}, {test_end}) - {test_end - test_start} samples\n')

    load_data_input = Load_Data(input_path=input_path, output_path=input_path)
    load_data_output = Load_Data(input_path=input_path, output_path=output_path)
    
    if mode in ['train', 'train_test']:
        print('==> Loading training and validation data...')
        train_x = load_data_input.load_data_by_indices(train_start, train_split)
        train_y = load_data_output.load_data_by_indices(train_start, train_split)
        val_x = load_data_input.load_data_by_indices(val_start, val_end)
        val_y = load_data_output.load_data_by_indices(val_start, val_end)
    
    if mode in ['test', 'train_test']:
        print('==> Loading test data...')
        test_x = load_data_input.load_data_by_indices(test_start, test_end)
        test_y = load_data_output.load_data_by_indices(test_start, test_end)

    input_dim = 6
    output_dim = 20

    # 创建ViT模型
    net = ViT3D(input_dim, output_dim, embed_dim=embed_dim, depth=depth, 
                num_heads=num_heads, patch_size=(2, 3, 3))

    if mode in ['train', 'train_test']:
        print('\n' + '='*50)
        print('Starting Training...')
        print('='*50)
        trainer = Train(net, epochs, lr, batch_size, filename, train_x, train_y, 
                       val_x, val_y, save_dir, patience, gpu_ids, 
                       accumulation_steps=accumulation_steps)
        trainer.train()
        
        show_loss = Show_Loss(filename, save_dir)
        show_loss.loss_image_save()
        print('\n==> Training loss curve saved!')

    if mode in ['test', 'train_test']:
        print('\n' + '='*50)
        print('Starting Testing...')
        print('='*50)
        
        result_dir = os.path.join(save_dir, 'test_results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        pred_path = os.path.join(result_dir, 'predictions_denorm.npy')
        test_y_path = os.path.join(result_dir, 'labels_denorm.npy')
        
        print(f"\n==> Running complete test with denormalization and mask...")
        print(f"Mask file: {output_mask_path}")
        
        if not os.path.exists(output_mask_path):
            print(f"Warning: Mask file not found at {output_mask_path}")
            print("Proceeding without mask application.")
        
        model_result = Model_Result(net, save_dir, batch_size, test_x, test_y, 
                                   output_mask_path, pred_path, test_y_path, 
                                   gpu_ids, num_workers=8)
        model_result.result()
        
        print("\n==> Calculating metrics on denormalized data...")
        metrics_calculator = Metrics(test_y_path, pred_path)
        r, r2, rmse, mae = metrics_calculator.calculate_all_metrics()
        
        metrics = {
            'Correlation': r,
            'R2_Score': r2,
            'RMSE': rmse,
            'MAE': mae
        }
        df_metrics = pd.DataFrame([metrics])
        metrics_path = os.path.join(result_dir, 'test_metrics_denorm.csv')
        df_metrics.to_csv(metrics_path, index=False)
        print(f"\n==> Metrics saved to: {metrics_path}")
        print(f"==> Predictions saved to: {pred_path}")
        print(f"==> Labels saved to: {test_y_path}")
        
        print("\n" + "="*50)
        print("Testing completed successfully!")
        print("="*50)


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Vision Transformer (ViT) Training and Testing')
    parser.add_argument('--mode', type=str, default='train_test', 
                       choices=['train', 'test', 'train_test'],
                       help='Mode: train, test, or train_test (default: train_test)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU IDs to use, e.g., "0" or "0,1,2" (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32, smaller than CNN due to memory)')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps (default: 1, use 2-4 to simulate larger batch)')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001, smaller than CNN)')
    parser.add_argument('--patience', type=int, default=8,
                       help='Early stopping patience (default: 8)')
    parser.add_argument('--mask_path', type=str, 
                       default=os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'output_mask.npy',
                       help='Path to output mask file')
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='Transformer embedding dimension (default: 256)')
    parser.add_argument('--depth', type=int, default=6,
                       help='Number of Transformer layers (default: 6)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    
    args = parser.parse_args()
    
    gpu_ids = [int(x) for x in args.gpu.split(',')]
    
    print("\n" + "="*70)
    print("Vision Transformer (ViT) Model - Training and Testing")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accumulation steps: {args.accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Patience: {args.patience}")
    print(f"Mask path: {args.mask_path}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Depth: {args.depth}")
    print(f"Num heads: {args.num_heads}")
    print("="*70 + "\n")
    
    main(mode=args.mode, 
         gpu_ids=gpu_ids, 
         batch_size=args.batch_size,
         epochs=args.epochs,
         lr=args.lr,
         patience=args.patience,
         use_denormalize=True,
         output_mask_path=args.mask_path,
         embed_dim=args.embed_dim,
         depth=args.depth,
         num_heads=args.num_heads,
         accumulation_steps=args.accumulation_steps)
