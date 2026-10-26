#!/usr/bin/env python
# coding: utf-8

"""
ConvLSTM模型训练和测试脚本

代码组织结构：
================================================================================
1. 导入和配置 (第18-35行)
2. 模型定义 (第38-200行)
   - ConvLSTMCell: LSTM单元
   - ConvLSTM: 多层LSTM网络
   - ConvLSTMModel: 完整模型
   
3. 工具类 (第203-275行)
   - random_seed: 设置随机种子
   - EarlyStopping: 早停机制
   
4. 数据处理类 (第278-350行)
   - CustomDataset: 自定义数据集
   - Load_Data: 数据加载器
   
5. 训练相关类 (第353-610行)
   - Train: 模型训练
   - Show_Loss: 损失曲线可视化
   
6. 测试相关类 (第613-880行)
   - Test: 简单测试（归一化数据）
   - Model_Result: 完整测试（反归一化+掩码）
   - Metrics: 评估指标计算
   
7. 主函数 (第883-1020行)
   - main: 主执行函数
   
8. 命令行接口 (第1023-1080行)
================================================================================
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非图形后端，必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")


# ============================================================================
# 模型定义
# ============================================================================

class ConvLSTMCell(nn.Module):
    """ConvLSTM单元"""
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # 保证在传递过程中 (h,w)不变
        
        # 修复：使用GroupNorm替代LayerNorm，避免固定空间尺寸
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim * 4,  # i门，f门，o门，g门放在一起计算，然后split开
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        # 使用GroupNorm替代LayerNorm，更节省显存且不依赖空间尺寸
        self.norm = nn.GroupNorm(num_groups=4, num_channels=self.hidden_dim * 4)

    def forward(self, input_tensor, cur_state):
        h_t, c_t = cur_state
        combined = torch.cat([input_tensor, h_t], dim=1)  # 把输入张量与h状态张量沿通道维度串联
        combined_conv = self.conv(combined)  # i门，f门，o门，g门放在一起计算，然后split开
        combined_conv = self.norm(combined_conv)  # 应用归一化
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_new = f * c_t + i * g  # c状态张量更新
        h_new = o * torch.tanh(c_new)  # h状态张量更新
        
        return h_new, c_new  # 输出当前timestamp的两个状态张量

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        init_h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        init_c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return (init_h, init_c)


class ConvLSTM(nn.Module):
    """ConvLSTM模型"""
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        
        # 确保kernel_size和hidden_dim是列表
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim):
            raise ValueError('Inconsistent list length.')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        input_tensor: (batch, channels, seq_len, height, width)
        """
        batch_size, _, seq_len, height, width = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width), input_tensor.device)
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                x = cur_layer_input[:, :, t, :, :]
                h, c = self.cell_list[layer_idx](x, (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=2)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        output = layer_output_list[-1]
        return output

    def _init_hidden(self, batch_size, image_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMModel(nn.Module):
    """
    完整的ConvLSTM模型
    输入: (batch, input_dim, depth, height, width)
    输出: (batch, output_dim, depth, height, width)
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2):
        super(ConvLSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # ConvLSTM层
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=[hidden_dim] * num_layers,
            kernel_size=(3, 3),
            num_layers=num_layers
        )
        
        # 输出卷积层，将hidden_dim映射到output_dim
        self.conv_out = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        """
        x: (batch, input_dim, depth, height, width)
        output: (batch, output_dim, depth, height, width)
        """
        batch_size, input_dim, depth, height, width = x.shape
        
        # ConvLSTM处理每个深度层
        # 将depth维度作为序列长度
        # 重排为 (batch, input_dim, depth, height, width)
        lstm_out = self.convlstm(x)  # (batch, hidden_dim, depth, height, width)
        
        # 对每个深度层应用输出卷积
        output = []
        for t in range(depth):
            out_t = self.conv_out(lstm_out[:, :, t, :, :])  # (batch, output_dim, height, width)
            output.append(out_t)
        
        output = torch.stack(output, dim=2)  # (batch, output_dim, depth, height, width)
        
        return output


# ============================================================================
# 工具类
# ============================================================================

class random_seed():
    
    def __init__(self,seed = 2023, benchmark = False, deterministic = False):
        
        self.seed = seed
        self.benchmark = benchmark
        self.deterministic = deterministic
        
    def set_seed(self):
        
        torch.manual_seed(self.seed)  # 固定随机种子（CPU）
        if torch.cuda.is_available():  # 固定随机种子（GPU)
            torch.cuda.manual_seed(self.seed)  # 为当前GPU设置
            torch.cuda.manual_seed_all(self.seed)  # 为所有GPU设置
            
        np.random.seed(self.seed)  # 保证后续使用random函数时，产生固定的随机数
        
        torch.backends.cudnn.benchmark = self.benchmark  # GPU、网络结构固定，可设置为True
        torch.backends.cudnn.deterministic = self.deterministic  # 固定网络结构（网络固定后再设置为True，可加快训练）



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
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
# 数据处理类
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
        # 确保数据类型为float32
        return data.astype(np.float32)

class Load_Data():
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def get_list_by_indices(self, file_path, start_idx, end_idx):
        """根据索引范围获取文件列表"""
        # 获取所有文件并排序
        files = sorted(os.listdir(file_path), 
                      key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # 只选择指定索引范围内的文件
        selected_files = []
        for file in files:
            # 提取文件名中的数字部分
            file_num = int(file.split('_')[1].split('.')[0])
            if start_idx <= file_num < end_idx:
                selected_files.append(file)
        
        # 生成完整路径列表
        data_list = [os.path.join(file_path, f) for f in selected_files]
        return data_list
    
    def get_list_by_ratio(self, file_path, start_seg, end_seg):
        """根据比例获取文件列表（保持原功能）"""
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
        
        # 把所有样本加载成一个大 numpy 数组，确保为float32类型
        all_data = np.stack([data_y[i] for i in range(len(data_y))], axis=0).astype(np.float32)
        return all_data
    
    def load_data_by_ratio(self, start_seg, end_seg, custom_transform=None):
        """根据比例加载数据（保持原功能）"""
        data_y_list = self.get_list_by_ratio(self.output_path, start_seg, end_seg)
        data_y = CustomDataset(data_y_list, transform=custom_transform)
        
        all_data = np.stack([data_y[i] for i in range(len(data_y))], axis=0)
        return all_data

# ============================================================================
# 训练相关类
# ============================================================================

class Train():
    
    def __init__(self, net, epochs, lr, batch_size, filename, train_x, train_y, val_x, val_y, save_dir, patience, gpu_ids, num_workers = 8, test_x=None, test_y=None):
        
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
        self.num_workers = num_workers   #num_workers用于数据加载中调用gpu的核心数
        
        
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
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # if torch.cuda.device_count() > 1:
        #     self.net = nn.DataParallel(self.net)
        # self.net.to(device)
        
                # 修改设备指定方式
        if len(self.gpu_ids) > 1:
            # 指定使用哪些GPU
            # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            # 将模型放到第一个GPU上
            self.net.to(device)
            # 使用DataParallel包装模型，指定设备ID
            self.net = nn.DataParallel(self.net, device_ids=list(range(len(self.gpu_ids))))
        else:
            # 如果只使用一个GPU
            device = torch.device(f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
            self.net.to(device)

        lossfunction = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,  weight_decay=1e-5)   #L2正则化
        
        if os.path.exists(os.path.join(self.save_dir, 'checkpoint.pth')):
            # load existing model
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


        train_losses = []        # to track the training loss as the model trains
        valid_losses = []        # to track the validation loss as the model trains
        avg_train_losses = []    # to track the average training loss per epoch as the model trains
        avg_valid_losses = []    # to track the average validation loss per epoch as the model trains
        # mini_val_loss = np.inf

        since = time.time()  # 当前时间

        for epoch in range(cur_epoch, self.epochs + 1):

            ###################
            # train the model #
            ###################
            #数据封装到数据通道中
            train_X = torch.utils.data.DataLoader(self.train_x, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            train_Y = torch.utils.data.DataLoader(self.train_y, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

            # t = tqdm(zip(train_X,train_Y), leave=False, total=len(train_X),colour = 'pink')
            t = tqdm(zip(train_X,train_Y), leave=False, total=len(train_X))
            
            for inputVar,targetVar in t:

                inputs = inputVar.to(device)  # B,S,C,H,W
                label = targetVar.to(device)  # B,S,C,H,W
                optimizer.zero_grad()
                self.net.train()
                pred = self.net(inputs)  # B,S,C,H,W
                #print(pred.shape)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / self.batch_size
                train_losses.append(loss_aver)
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_value=10.0)
                optimizer.step()
                t.set_postfix({
                    'trainloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })
            
            #TensorBoard添加信息
            tb.add_scalar('TrainLoss', loss_aver, epoch)
            ######################
            # validate the model #
            ######################
            with torch.no_grad():
                self.net.eval()

                val_X = torch.utils.data.DataLoader(self.val_x, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                val_Y = torch.utils.data.DataLoader(self.val_y, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

                t = tqdm(zip(val_X,val_Y), leave=False, total=len(val_X),colour = 'pink')
                for inputVar,targetVar in t:

                    inputs = inputVar.to(device)
                    label = targetVar.to(device)
                    pred = self.net(inputs)
                    loss = lossfunction(pred, label)        #label.shape[16, 12, 13, 46, 91]
                    loss_aver = loss.item() / self.batch_size
                    # record validation loss
                    valid_losses.append(loss_aver)
                    #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                    t.set_postfix({
                        'validloss': '{:.6f}'.format(loss_aver),
                        'epoch': '{:02d}'.format(epoch)
                    })

                    
            #TensorBoard添加信息
            tb.add_scalar('ValidLoss', loss_aver, epoch)
            
            #清空显存缓冲区
            torch.cuda.empty_cache()
            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(self.epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.6f} ' +
                         f'valid_loss: {valid_loss:.6f}')

            print(print_msg)
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            pla_lr_scheduler.step(valid_loss)  # lr_scheduler
            model_dict = {
                'epoch': epoch,
                'state_dict': self.net.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            early_stopping(valid_loss, model_dict, epoch, self.save_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    #     time_use = time.time() - since  # 计算耗费时间    
        self.print_use_time(time.time() - since)  # 计算耗费时间    

        #save loss
        loss_data = {
            'avg_train_loss':avg_train_losses,
            'avg_valid_loss':avg_valid_losses
        }
        df_loss = pd.DataFrame(loss_data)

        loss_dir = os.path.join(self.save_dir, 'loss')
        
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
            
        df_loss.to_csv(os.path.join(loss_dir, self.filename+'.csv'))


# %%
class Show_Loss():
    
    def __init__(self, filename, save_dir):
        
        self.filename = filename
        self.save_dir = save_dir
        
    def loss_image_save(self):
        # 读取 loss_csv 文件
        
        loss_dir = os.path.join(self.save_dir, 'loss')
        load_path = os.path.join(loss_dir, self.filename+'.csv')
        save_path = os.path.join(loss_dir, self.filename+'.jpg')
        
        df_loss = pd.read_csv(load_path, index_col=0)

        # 创建一个画布
        fig = plt.figure(figsize = (9,5),dpi = 80)
        # 在画布上添加一个子图像（1行，1列，子图像索引为1）
        ax = fig.add_subplot(1, 1, 1)


        # 绘制折线图
        ax.plot(df_loss.avg_train_loss, label='train_loss')
        ax.plot(df_loss.avg_valid_loss, label='valid_loss')

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)

        # 添加标题和标签
        ax.set_title('Training and Validation Loss Over Time')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        # 添加图例
        ax.legend()
        
        plt.savefig(save_path,bbox_inches = 'tight',dpi=100)


# ============================================================================
# 测试相关类
# ============================================================================

class Test():
    """
    模型测试类（归一化数据）
    用于在测试集上评估模型性能，不进行反归一化处理
    """
    
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
        
        # 设置设备
        if len(self.gpu_ids) > 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {device}")
        
        # 加载模型
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 处理DataParallel的module.前缀
        if any(k.startswith('module.') for k in state_dict.keys()):
            print("Removing 'module.' prefix from checkpoint...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 加载模型权重
        self.net.to(device)
        self.net.load_state_dict(state_dict, strict=False)
        
        # 多GPU支持
        if len(self.gpu_ids) > 1 and torch.cuda.is_available():
            self.net = nn.DataParallel(self.net, device_ids=list(range(len(self.gpu_ids))))
        
        self.net.eval()
        
        # 准备数据加载器
        test_X = torch.utils.data.DataLoader(self.test_x, batch_size=self.batch_size, 
                                            shuffle=False, num_workers=self.num_workers)
        test_Y = torch.utils.data.DataLoader(self.test_y, batch_size=self.batch_size, 
                                            shuffle=False, num_workers=self.num_workers)
        
        # 测试
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
                
                # 保存预测结果
                pre_list.append(pred.cpu().numpy())
                label_list.append(targetVar.numpy())
                
                t.set_postfix({'test_loss': '{:.6f}'.format(loss_aver)})
        
        # 合并所有批次的结果
        predictions = np.concatenate(pre_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        
        # 保存结果
        result_dir = os.path.join(self.save_dir, 'test_results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        np.save(os.path.join(result_dir, 'predictions.npy'), predictions)
        np.save(os.path.join(result_dir, 'labels.npy'), labels)
        
        # 保存测试损失
        avg_test_loss = np.mean(test_losses)
        loss_df = pd.DataFrame({'test_loss': test_losses})
        loss_df.to_csv(os.path.join(result_dir, 'test_loss.csv'), index=False)
        
        print(f'\n==> Test completed!')
        print(f'Average test loss: {avg_test_loss:.6f}')
        print(f'Results saved to: {result_dir}')
        
        # 计算基本指标
        r2 = self._calculate_r2(labels, predictions)
        rmse = self._calculate_rmse(labels, predictions)
        mae = self._calculate_mae(labels, predictions)
        r = self._calculate_correlation(labels, predictions)
        
        print(f'\n==> Test Metrics (Normalized Data):')
        print(f'Correlation: {r:.4f}')
        print(f'R2 Score: {r2:.4f}')
        print(f'RMSE: {r2:.4f}')
        print(f'MAE: {mae:.4f}')
        
        # 保存指标
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
        """计算R2分数"""
        mse = np.mean((y_true - y_pred) ** 2)
        var = np.var(y_true)
        return 1 - mse / var if var != 0 else 0
    
    def _calculate_rmse(self, y_true, y_pred):
        """计算RMSE"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _calculate_mae(self, y_true, y_pred):
        """计算MAE"""
        return np.mean(np.abs(y_true - y_pred))
    
    def _calculate_correlation(self, y_true, y_pred):
        """计算相关系数"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return np.corrcoef(y_true_flat, y_pred_flat)[0, 1]


class Model_Result():
    """
    模型结果处理类（反归一化+掩码）
    用于生成预测结果、反归一化和应用掩码，得到真实物理值
    """
    
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
        
        # 归一化常数
        min_lst = [0.0579, 0.0571, 0.0548, 0.0513, 0.0474, 0.046, 0.0504, 0.0617, 
                   0.068, 0.0587, 0.0472, 0.0381, 0.0316, 0.0261, 0.0214, 0.0182, 
                   0.0137, 0.011, 0.0075, 0.0051]
        dif_lst = [0.7557, 0.7907, 0.7054, 0.6156, 0.5476, 0.4894, 0.4408, 0.3896, 
                   0.3514, 0.3359, 0.3265, 0.3162, 0.2894, 0.269, 0.2483, 0.2166, 
                   0.1421, 0.0878, 0.0332, 0.0161]
        
        # 转为可广播的形状 (1, 20, 1, 1, 1)
        min_arr = np.array(min_lst).reshape(1, -1, 1, 1, 1)
        dif_arr = np.array(dif_lst).reshape(1, -1, 1, 1, 1)
        
        # 反归一化: original = normalized * diff + min
        return data * dif_arr + min_arr

    def set_mask(self, data, mask_file):
        """应用掩码"""
        data_length = data.shape[0]
        # 将mask_file扩展到与data相同的batch维度
        mask = np.tile(mask_file, (data_length, 1, 1, 1, 1))
        
        # 确保mask是布尔类型
        # 如果mask_file中用NaN标记无效区域，则转换为布尔掩码
        if np.isnan(mask).any():
            mask = np.isnan(mask)
        else:
            # 如果mask_file用其他值（如0/1），转换为布尔类型
            mask = mask.astype(bool)
        
        # 应用掩码
        data[mask] = np.nan
        return data
        
    def result(self):
        """执行完整的测试流程：推理 -> 反归一化 -> 掩码 -> 保存"""
        
        # 设置设备
        if len(self.gpu_ids) > 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {device}, CUDA count: {torch.cuda.device_count()}")
        
        # 加载checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        epoch = checkpoint.get('epoch', None)
        
        print(f"Loaded checkpoint from epoch: {epoch}")
        
        # 处理module.前缀
        if any(k.startswith('module.') for k in state_dict.keys()):
            print("Removing 'module.' prefix from checkpoint...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 加载模型
        self.net.to(device)
        self.net.load_state_dict(state_dict, strict=False)
        
        # 多GPU支持
        if len(self.gpu_ids) > 1 and torch.cuda.is_available():
            print(f"Using {len(self.gpu_ids)} GPUs with DataParallel...")
            self.net = nn.DataParallel(self.net, device_ids=list(range(len(self.gpu_ids))))
        
        self.net.eval()
        
        # 加载数据
        test_X = torch.utils.data.DataLoader(self.test_x, batch_size=self.batch_size, 
                                            shuffle=False, num_workers=self.num_workers)
        test_Y = torch.utils.data.DataLoader(self.test_y, batch_size=self.batch_size, 
                                            shuffle=False, num_workers=self.num_workers)
        
        # 推理
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
        
        # 反归一化
        print("Applying denormalization...")
        pred = self.output_inver(pre)
        test_y = self.output_inver(label)
        
        # 应用掩码
        if self.output_mask_path and os.path.exists(self.output_mask_path):
            print(f"Applying mask from: {self.output_mask_path}")
            output_mask = np.load(self.output_mask_path)
            pred = self.set_mask(pred, output_mask)
            test_y = self.set_mask(test_y, output_mask)
        else:
            print("Warning: No mask file provided or file not found. Skipping mask application.")
        
        # 保存结果
        print(f"Saving results...")
        np.save(self.pred_path, pred)
        np.save(self.test_y_path, test_y)
        
        print(f"✅ Results saved:")
        print(f"  - Predictions: {self.pred_path}")
        print(f"  - Labels: {self.test_y_path}")





class Metrics():
    """
    评估指标计算类
    从保存的文件中加载预测结果和真实标签，计算各种评估指标
    支持处理包含NaN值的数据（用于掩码后的数据）
    """
    
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
        # 展开为一维数组
        test_y_flat = test_y.flatten()
        pred_flat = pred.flatten()
        
        # 移除NaN值
        nan_mask = np.isnan(test_y_flat) | np.isnan(pred_flat)
        test_y_clean = test_y_flat[~nan_mask]
        pred_clean = pred_flat[~nan_mask]
        
        if len(test_y_clean) == 0:
            print("Warning: No valid data points for correlation calculation")
            return np.nan
        
        # 计算相关系数
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
        
        # 统计有效数据点数量
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

def main(mode='train_test', batch_size=32, epochs=300, lr=0.001, patience=8, 
         use_denormalize=True, output_mask_path=os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'output_mask.npy'):
    """
    主函数
    mode: 'train' - 仅训练
          'test' - 仅测试
          'train_test' - 训练后测试（默认）
    batch_size: 批次大小
    epochs: 训练轮数
    lr: 学习率
    patience: 早停耐心值
    use_denormalize: 是否使用反归一化和掩码（使用Model_Result类），默认True
    output_mask_path: 掩码文件路径，默认为你的掩码文件位置
    
    注意：使用 CUDA_VISIBLE_DEVICES 环境变量指定 GPU
    例如：CUDA_VISIBLE_DEVICES=0,1,2 python ConvLSTM.py --mode train_test
    """
    
    # 自动检测可用的 GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_ids = list(range(gpu_count))
        print(f'\n==> 检测到 {gpu_count} 个可用 GPU: {gpu_ids}')
    else:
        gpu_ids = []
        print('\n==> 未检测到 GPU，将使用 CPU')
    
    input_path = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'pre_train_data/input/'
    output_path = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'pre_train_data/output/'

    filename = 'ConvLSTM'
    save_dir = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + '' + filename

    print(f'\n==> Configuration:')
    print(f'Mode: {mode}')
    print(f'Available GPUs: {gpu_ids if gpu_ids else "CPU only"}')
    print(f'Batch size: {batch_size}')
    print(f'Epochs: {epochs}')
    print(f'Learning rate: {lr}')
    print(f'Patience: {patience}')
    print(f'Use denormalize: {use_denormalize}')

    #设置随机种子
    set_seeds = random_seed(seed=2023, benchmark=False, deterministic=False)  #后两个参数等模型固定再设置为True
    set_seeds.set_seed()

    # 计算数据集划分点
    train_start = 1827
    train_end = 7671
    train_total = train_end - train_start  # 5844个样本
    
    # 训练集：前90% (1827 - 7086)
    train_split = train_start + int(train_total * 0.9)  # 7086
    
    # 验证集：后10% (7086 - 7671)
    val_start = train_split
    val_end = train_end
    
    # 测试集：(7671 - 9119)
    test_start = 7671
    test_end = 9119

    print(f'\n==> Data split:')
    print(f'Training set: [{train_start}, {train_split}) - {train_split - train_start} samples')
    print(f'Validation set: [{val_start}, {val_end}) - {val_end - val_start} samples')
    print(f'Test set: [{test_start}, {test_end}) - {test_end - test_start} samples\n')

    # 加载数据
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

    # 模型参数
    input_dim = 6
    output_dim = 20
    hidden_dim = 64  # ConvLSTM隐藏层维度
    num_layers = 2   # ConvLSTM层数

    # 创建模型
    net = ConvLSTMModel(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    
    print(f'\n==> Model Configuration:')
    print(f'Input dimension: {input_dim}')
    print(f'Output dimension: {output_dim}')
    print(f'Hidden dimension: {hidden_dim}')
    print(f'Number of LSTM layers: {num_layers}')

    # 训练模式
    if mode in ['train', 'train_test']:
        print('\n' + '='*50)
        print('Starting Training...')
        print('='*50)
        trainer = Train(net, epochs, lr, batch_size, filename, train_x, train_y, 
                       val_x, val_y, save_dir, patience, gpu_ids)
        trainer.train()
        
        # 显示训练损失曲线
        show_loss = Show_Loss(filename, save_dir)
        show_loss.loss_image_save()
        print('\n==> Training loss curve saved!')

    # 测试模式
    if mode in ['test', 'train_test']:
        print('\n' + '='*50)
        print('Starting Testing...')
        print('='*50)
        
        # 默认使用完整测试流程（反归一化+掩码）
        result_dir = os.path.join(save_dir, 'test_results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        pred_path = os.path.join(result_dir, 'predictions_denorm.npy')
        test_y_path = os.path.join(result_dir, 'labels_denorm.npy')
        
        print(f"\n==> Running complete test with denormalization and mask...")
        print(f"Mask file: {output_mask_path}")
        
        # 检查掩码文件是否存在
        if not os.path.exists(output_mask_path):
            print(f"Warning: Mask file not found at {output_mask_path}")
            print("Proceeding without mask application.")
        
        # 使用Model_Result进行完整测试
        model_result = Model_Result(net, save_dir, batch_size, test_x, test_y, 
                                   output_mask_path, pred_path, test_y_path, 
                                   gpu_ids, num_workers=8)
        model_result.result()
        
        # 使用Metrics类计算指标
        print("\n==> Calculating metrics on denormalized data...")
        metrics_calculator = Metrics(test_y_path, pred_path)
        r, r2, rmse, mae = metrics_calculator.calculate_all_metrics()
        
        # 保存指标
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
    
    parser = argparse.ArgumentParser(
        description='ConvLSTM Training and Testing',
        epilog='使用 CUDA_VISIBLE_DEVICES 环境变量指定 GPU，例如：\n'
               '  CUDA_VISIBLE_DEVICES=0 python ConvLSTM.py --mode train_test\n'
               '  CUDA_VISIBLE_DEVICES=0,1,2 python ConvLSTM.py --mode train_test',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mode', type=str, default='train_test', 
                       choices=['train', 'test', 'train_test'],
                       help='Mode: train, test, or train_test (default: train_test)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16, 优化后的值)')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=8,
                       help='Early stopping patience (default: 8)')
    parser.add_argument('--mask_path', type=str, 
                       default=os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'output_mask.npy',
                       help='Path to output mask file')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ConvLSTM Model - Training and Testing")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Patience: {args.patience}")
    print(f"Mask path: {args.mask_path}")
    
    # 显示 GPU 信息
    if torch.cuda.is_available():
        print(f"\n检测到的 GPU:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("\n未检测到 GPU，将使用 CPU")
    
    print("="*70 + "\n")
    
    main(mode=args.mode, 
         batch_size=args.batch_size,
         epochs=args.epochs,
         lr=args.lr,
         patience=args.patience,
         use_denormalize=True,
         output_mask_path=args.mask_path)


# %%

