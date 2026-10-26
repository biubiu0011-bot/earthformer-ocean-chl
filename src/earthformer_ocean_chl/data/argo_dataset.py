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
                 data_range=(0, 2909, 1), train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """
        Args:
            data_range: tuple of (start, end, step) for data indices
            train_ratio: 训练集比例，默认 0.7 (70%)
            val_ratio: 验证集比例，默认 0.1 (10%)
            test_ratio: 测试集比例，默认 0.2 (20%)
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.split = split
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # 生成完整的数据索引范围
        all_indices = np.arange(data_range[0], data_range[1], data_range[2])
        total_samples = len(all_indices)
        
        # 计算各个数据集的大小（确保三者加起来等于 100%）
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        # test_size 使用剩余的所有数据，确保没有数据丢失
        
        # 顺序划分数据集：训练集 -> 验证集 -> 测试集
        train_end = train_size
        val_end = train_size + val_size
        
        train_indices = all_indices[:train_end]
        val_indices = all_indices[train_end:val_end]
        test_indices = all_indices[val_end:]
        
        print(f"数据集顺序划分完成:")
        print(f"  总样本数: {total_samples}")
        print(f"  训练集: {len(train_indices)} 个样本 ({len(train_indices)/total_samples*100:.1f}%) - 索引 {train_indices[0]}-{train_indices[-1]}")
        print(f"  验证集: {len(val_indices)} 个样本 ({len(val_indices)/total_samples*100:.1f}%) - 索引 {val_indices[0]}-{val_indices[-1]}")
        print(f"  测试集: {len(test_indices)} 个样本 ({len(test_indices)/total_samples*100:.1f}%) - 索引 {test_indices[0]}-{test_indices[-1]}")
        
        # 保存所有三个数据集的索引
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        # 根据 train 参数决定使用哪个数据集
        # train=True 时返回训练集，train=False 时返回测试集
        # 验证集通过 DataModule 访问
        if self.train:
            self.data_indices = train_indices
        else:
            self.data_indices = test_indices

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

        # 使用统一的 data_indices
        data_idx = self.data_indices[index]
        seq = np.nan_to_num(np.array(np.load(pretrain_path('transfer_input', 'input_'+str(data_idx)+'.npy')), dtype=np.float32).transpose(1,2,3,0), nan=0.0)
        target = np.nan_to_num(np.array(np.load(pretrain_path('transfer_output', 'output_' + str(data_idx) + '.npy')), dtype=np.float32).transpose(1, 2, 3, 0), nan=0.0)

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
        return len(self.data_indices)
       

    # def _check_exists(self):
    #     return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
    #         os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
    # 修改
    def _check_exists(self):
        input_dir = pretrain_path('transfer_input')
        output_dir = pretrain_path('transfer_output')

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
                 data_range=(0, 2909, 1), train_ratio=0.7, test_ratio=0.2):
        """

        Parameters
        ----------
        root
        val_ratio
            验证集比例（从训练集中划分），默认 0.1
        batch_size
        rescale_input_shape
            For the purpose of testing. Rescale the inputs
        rescale_target_shape
            For the purpose of testing. Rescale the targets
        data_range
            数据索引范围 (start, end, step)
        train_ratio
            训练集比例，默认 0.7
        test_ratio
            测试集比例，默认 0.2
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
            # 创建训练数据集（train=True 返回训练集）
            train_dataset = MovingMNIST(
                self.root, 
                train=True,
                post_transform=self.post_transform,
                post_target_transform=self.post_target_transform,
                data_range=self.data_range,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio
            )
            
            # 创建验证数据集（使用相同的 dataset 对象，但通过索引访问验证集部分）
            # 由于验证集索引已经在 __init__ 中划分好了，我们创建一个 Subset 来访问
            val_indices = list(range(len(train_dataset.val_indices)))
            
            # 创建一个临时的验证集 dataset
            val_dataset = MovingMNIST(
                self.root, 
                train=True,  # 先设为 True 以访问完整数据
                post_transform=self.post_transform,
                post_target_transform=self.post_target_transform,
                data_range=self.data_range,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio
            )
            # 手动设置为验证集索引
            val_dataset.data_indices = val_dataset.val_indices
            
            self.moving_mnist_train = train_dataset
            self.moving_mnist_val = val_dataset
            
            print(f"✓ 数据集加载完成（完全顺序划分）:")
            print(f"  训练集: {len(self.moving_mnist_train)} 个样本")
            print(f"  验证集: {len(self.moving_mnist_val)} 个样本")

        if stage == "test" or stage is None:
            self.moving_mnist_test = MovingMNIST(
                self.root, 
                train=False,
                post_transform=self.post_transform,
                post_target_transform=self.post_target_transform,
                data_range=self.data_range,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio
            )

        if stage == "predict" or stage is None:
            self.moving_mnist_predict = MovingMNIST(
                self.root, 
                train=False,
                post_transform=self.post_transform,
                post_target_transform=self.post_target_transform,
                data_range=self.data_range,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio
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
