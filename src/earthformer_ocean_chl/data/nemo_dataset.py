"""Code is adapted from https://github.com/tychovdo/MovingMNIST."""

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import torch.nn.functional as F
import codecs
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Subset, DataLoader
from typing import Optional
from .temporal_augmentation import TemporalAugmentation, DistributionAdaptiveAugmentation
from earthformer_ocean_chl.paths import pretrain_path


class MovingMNIST(data.Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None,
                 post_transform=None, post_target_transform=None, download=False,
                 data_range=(0, 9119, 1), train_ratio=0.8, test_ratio=0.2,
                 use_augmentation=False, augmentation_config=None):
        """
        Args:
            data_range: tuple of (start, end, step) for data indices
            train_ratio: 训练集比例，默认 0.8 (80%)
            test_ratio: 测试集比例，默认 0.2 (20%)
            use_augmentation: 是否使用数据增强（仅在训练时）
            augmentation_config: 数据增强配置字典
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.split = split
        self.train = train  # training set or test set
        self.use_augmentation = use_augmentation and train  # 只在训练时增强
        
        # 初始化数据增强
        if self.use_augmentation:
            if augmentation_config is None:
                augmentation_config = {}
            self.augmentation = TemporalAugmentation(**augmentation_config)
            print("✓ 数据增强已启用")
        else:
            self.augmentation = None

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # 生成完整的数据索引范围
        all_indices = np.arange(data_range[0], data_range[1], data_range[2])
        total_samples = len(all_indices)
        
        # 按时间顺序划分训练集和测试集（80%/20%）
        # 训练集：前 80% 的时间段
        # 测试集：后 20% 的时间段
        train_end = int(total_samples * train_ratio)
        
        train_indices = all_indices[:train_end]  # 前 80%
        test_indices = all_indices[train_end:]   # 后 20%
        
        print(f"数据集按时间顺序划分完成:")
        print(f"  总样本数: {total_samples}")
        print(f"  训练集: {len(train_indices)} 个样本 ({len(train_indices)/total_samples*100:.1f}%) - 索引范围 [{train_indices[0]}, {train_indices[-1]}]")
        print(f"  测试集: {len(test_indices)} 个样本 ({len(test_indices)/total_samples*100:.1f}%) - 索引范围 [{test_indices[0]}, {test_indices[-1]}]")
        
        # 保存训练集和测试集的索引
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        # 根据 train 参数决定使用哪个数据集
        if self.train:
            self.train_data = self.train_indices
        else:
            self.test_data = self.test_indices

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are split into a seq
                    and target part
        """

        # need to iterate over time
        def _transform_time(data):
            new_data = []
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data.append(self.transform(img))
            return torch.cat(new_data, dim=0)

        if self.train:
            seq = np.nan_to_num(np.array(np.load(pretrain_path('input_log_zscore', 'input_'+str(self.train_data[index])+'.npy')), dtype=np.float32).transpose(1,2,3,0), nan=0.0)
            target = np.nan_to_num(np.array(np.load(pretrain_path('2output_log_zscore_log10', 'output_' + str(self.train_data[index]) + '.npy')), dtype=np.float32).transpose(1, 2, 3, 0), nan=0.0)
        else:
            # print('------------------------', self.test_data[index])
            seq = np.nan_to_num(np.array(np.load(pretrain_path('input_log_zscore', 'input_'+str(self.test_data[index])+'.npy')), dtype=np.float32).transpose(1, 2, 3, 0), nan=0.0)
            target = np.nan_to_num(np.array(np.load(pretrain_path('2output_log_zscore_log10', 'output_' + str(self.test_data[index]) + '.npy')), dtype=np.float32).transpose(1, 2, 3, 0), nan=0.0)

        # 应用数据增强（仅在训练时）
        if self.use_augmentation and self.augmentation is not None:
            seq, target = self.augmentation(seq, target)

        if self.transform is not None:
            seq = _transform_time(seq)
        if self.target_transform is not None:
            target = _transform_time(target)
        # if self.post_transform is not None:
        #     seq = self.post_transform(seq)
        # if self.post_target_transform is not None:
        #     target = self.post_target_transform(target)
        # [10, 64, 64, 1] [10, 64, 64, 1]
        # print("------------------------", seq.shape, target.shape)
        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        

    # def _check_exists(self):
    #     return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
    #         os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
    # 修改
    def _check_exists(self):
        input_dir = pretrain_path('input_log_zscore')
        output_dir = pretrain_path('2output_log_zscore_log10')

        # 检查路径是否存在
        if not os.path.exists(input_dir) or not os.path.exists(output_dir):
            return False

        # 检查是否有实际数据文件存在
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]

        return len(input_files) > 0 and len(output_files) > 0

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # for url in self.urls:
        #     print('Downloading ' + url)
        #     data = urllib.request.urlopen(url)
        #     filename = url.rpartition('/')[2]
        #     file_path = os.path.join(self.root, self.raw_folder, filename)
        #     with open(file_path, 'wb') as f:
        #         f.write(data.read())
        #     with open(file_path.replace('.gz', ''), 'wb') as out_f, \
        #             gzip.GzipFile(file_path) as zip_f:
        #         out_f.write(zip_f.read())
        #     os.unlink(file_path)

        # # process and save as torch files
        # print('Processing...')
        #
        # training_set = torch.from_numpy(
        #     np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        # )
        # test_set = torch.from_numpy(
        #     np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        # )
        #
        # with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
        #     torch.save(training_set, f)
        # with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
        #     torch.save(test_set, f)
        #
        # print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class NearestInterpTransform:
    def __init__(self, target_thw, layout='THWC'):
        """

        Parameters
        ----------
        target_thw
            The target shape with (T, H, W)
        """
        self.target_thw = target_thw
        self.layout = layout

    def __call__(self, data):
        """

        Parameters
        ----------
        data
            Shape (T, H, W) or (T, H, W, C)

        Returns
        -------
        rescaled_data
            Shape (T, H, W, C)
            C will be 1 if the input data shape is (T, H, W)
        """
        if self.target_thw == data.shape:
            return data.view(*tuple(self.target_thw + (1,)))
        else:
            assert len(data.shape) == 3
            rescaled_data = F.interpolate(data.view((1, 1) + data.shape), self.target_thw, mode='nearest')
            rescaled_data = rescaled_data.view(self.target_thw + (1,))
            print('rescaled_data.shape=', rescaled_data.shape)
            return rescaled_data


class MovingMNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str = None,
                 val_ratio=0.1, seed=123, batch_size: int = 32,
                 rescale_input_shape=None, rescale_target_shape=None,
                 data_range=(0, 9119, 1), train_ratio=0.8, test_ratio=0.2,
                 use_augmentation=False, augmentation_config=None):
        """

        Parameters
        ----------
        root
        val_ratio
            验证集比例（从训练集中随机划分），默认 0.1
        seed
            随机种子
        batch_size
        rescale_input_shape
            For the purpose of testing. Rescale the inputs
        rescale_target_shape
            For the purpose of testing. Rescale the targets
        data_range
            数据索引范围 (start, end, step)，默认 (0, 9119, 1)
        train_ratio
            训练集比例（按时间顺序），默认 0.8 (80%)
        test_ratio
            测试集比例（按时间顺序），默认 0.2 (20%)
        use_augmentation
            是否使用数据增强，默认 False
        augmentation_config
            数据增强配置字典，例如:
            {
                'enable_noise': True,
                'enable_scaling': True,
                'enable_spatial_shift': True,
                'noise_std': 0.02,
                'scale_range': (0.95, 1.05),
                'shift_range': 2,
                'augment_prob': 0.5
            }
        """
        super().__init__()
        if root is None:
            from ...config import cfg
            root = os.path.join(cfg.datasets_dir, "moving_mnist")
        self.root = root
        self.val_ratio = val_ratio
        self.seed = seed
        self.batch_size = batch_size
        self.rescale_input_shape = rescale_input_shape
        self.rescale_target_shape = rescale_target_shape
        self.data_range = data_range
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config

        if self.rescale_input_shape is None:
            self.post_transform = NearestInterpTransform(target_thw=(10, 64, 64))
        else:
            self.post_transform = NearestInterpTransform(target_thw=self.rescale_input_shape)

        if self.rescale_target_shape is None:
            self.post_target_transform = NearestInterpTransform(target_thw=(10, 64, 64))
        else:
            self.post_target_transform = NearestInterpTransform(target_thw=self.rescale_target_shape)

    def prepare_data(self):
        # MovingMNIST(self.root, train=True, download=True)
        # MovingMNIST(self.root, train=False, download=True)
        pass

    @property
    def input_shape(self):
        """

        Returns
        -------
        ret
            Contains (T, H, W, C)
        """
        if self.rescale_input_shape is not None:
            return self.rescale_input_shape + (1,)
        else:
            return 14, 45, 60, 6

    @property
    def target_shape(self):
        """

        Returns
        -------

        """
        if self.rescale_target_shape is not None:
            return self.rescale_target_shape + (1,)
        else:
            return 14, 45, 60, 20

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # 创建训练数据集（已经在 __init__ 中完成了按时间顺序的 80%/20% 划分）
            train_val_data = MovingMNIST(
                self.root, 
                train=True,
                post_transform=self.post_transform,
                post_target_transform=self.post_target_transform,
                data_range=self.data_range,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                use_augmentation=self.use_augmentation,
                augmentation_config=self.augmentation_config
            )
            
            # 从训练集中随机划分出验证集（10%）
            all_indices = range(len(train_val_data))
            train_indices, val_indices = train_test_split(
                all_indices, 
                test_size=self.val_ratio, 
                random_state=self.seed,
                shuffle=True
            )
            
            self.moving_mnist_train = Subset(train_val_data, train_indices)
            self.moving_mnist_val = Subset(train_val_data, val_indices)
            
            print(f"\n✓ 训练集内部随机划分验证集完成:")
            print(f"  训练样本: {len(self.moving_mnist_train)} 个 ({len(train_indices)/len(train_val_data)*100:.1f}%)")
            print(f"  验证样本: {len(self.moving_mnist_val)} 个 ({len(val_indices)/len(train_val_data)*100:.1f}%)")
            print(f"\n最终数据分布:")
            print(f"  训练集: {len(self.moving_mnist_train)} 个样本 (总数据的 {len(self.moving_mnist_train)/(len(train_val_data)/(self.train_ratio))*100:.1f}%)")
            print(f"  验证集: {len(self.moving_mnist_val)} 个样本 (总数据的 {len(self.moving_mnist_val)/(len(train_val_data)/(self.train_ratio))*100:.1f}%)")
            print(f"  测试集: 将在测试阶段加载 (总数据的 {self.test_ratio*100:.1f}%)")
            if self.use_augmentation:
                print(f"  数据增强: 已启用")

        if stage == "test" or stage is None:
            self.moving_mnist_test = MovingMNIST(
                self.root, 
                train=False,
                post_transform=self.post_transform,
                post_target_transform=self.post_target_transform,
                data_range=self.data_range,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                use_augmentation=False  # 测试时不使用增强
            )

        if stage == "predict" or stage is None:
            self.moving_mnist_predict = MovingMNIST(
                self.root, 
                train=False,
                post_transform=self.post_transform,
                post_target_transform=self.post_target_transform,
                data_range=self.data_range,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                use_augmentation=False  # 预测时不使用增强
            )

    def train_dataloader(self):
        return DataLoader(self.moving_mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.moving_mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.moving_mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.moving_mnist_predict, batch_size=self.batch_size, shuffle=False, num_workers=4)

    @property
    def num_train_samples(self):
        return len(self.moving_mnist_train)

    @property
    def num_val_samples(self):
        return len(self.moving_mnist_val)

    @property
    def num_test_samples(self):
        return len(self.moving_mnist_test)

    @property
    def num_predict_samples(self):
        return len(self.moving_mnist_predict)
