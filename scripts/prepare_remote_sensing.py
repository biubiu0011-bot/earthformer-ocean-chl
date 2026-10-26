import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import notebook
import os

lat_interp = np.arange(10,55,1)
lon_interp = np.arange(120,180,1)

# depth_interp=np.array([15, 30, 45, 60, 80, 100, 120, 140, 160, 180, 200,250,299])
depth_interp=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 250 ,299])
# 创建完整日时间序列（2000-01-01 到 2024-12-31）
full_time = pd.date_range(start='2000-01-01', end='2024-12-31', freq='D')

input_mask = xr.open_dataset(os.path.join(os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data'), 'ocean_mask_depth0.nc'))
print(input_mask)

mask = input_mask['mask']

chla_ori = xr.open_mfdataset(os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'remote_data/CNR_chla/*.nc')
print(chla_ori)

chla_ori = chla_ori.sel(time=slice('2000','2024'))

# 自动将相同时间的数据取平均（处理重复最稳健的方式）
chla_clean = chla_ori.groupby('time').mean()

# 验证是否还有重复
print("清理后是否有重复时间？", 
      len(chla_clean.time) != len(np.unique(chla_clean.time.values)))
print(chla_clean)

chla = chla_clean.CHL.interp(latitude = lat_interp,longitude = lon_interp)

chla_masked = chla.where(~mask.isnull())  # mask 为 NaN 的位置变成 NaN
print('begin to save')

chla_masked.to_netcdf(os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'remote_data/remote_data_merge/remote_interp_mask/chla_00-24interp_masked.nc')
print('sussccesful')
