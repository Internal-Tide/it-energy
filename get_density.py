#!/public/home/lfs/Soft/miniconda3/envs/tzw/bin/python3
# This script uses MPI to calculate seawater density in parallel using the TEOS-10 GSW package
import os
import numpy as np
import xarray as xr
import gsw
from mpi4py import MPI

# MPI 初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# define the year and month
year = 2016
month = 8
days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
file_path = f"/data/lfs/Tide/Data2016/TS{year}{month:02d}"

# 仅在 rank 0 上获取文件列表，并分发给各个进程
if rank == 0:
    filenames = []
    for i in range(days_in_month[month-1]):
        for j in range(24):
            filename = f'{file_path}/ZIP_TSHRMMEAN{year}-{month:02d}-{i+1:02d}-{j+1:02d}.nc'
            if os.path.exists(filename):
                filenames.append(filename)
            else:
                print(f'{filename} does not exist')
    # 将任务按进程数均分
    tasks = [filenames[i::size] for i in range(size)]
else:
    tasks = None

# 分发任务，每个进程获得一个文件列表
local_filenames = comm.scatter(tasks, root=0)

# 每个进程处理自己分到的文件
dens_path = f'/data/lfs/Tide/result2016/dens{year}{month:02d}'
if not os.path.exists(dens_path):
    os.makedirs(dens_path, exist_ok=True)

for filename in local_filenames:
    print(f"Rank {rank} processing {filename}")
    ds = xr.open_dataset(filename)
    # calculate the density
    pt = ds['tshour'].values[0,:,:,:]
    ps = ds['sshour'].values[0,:,:,:]
    p = gsw.p_from_z(ds['lev1'].values.reshape((55,1,1)),
                      ds['lat'].values.reshape((1,2302,3600)))
    sa = gsw.SA_from_SP(ps, p,
                        ds['lon'].values.reshape((1,2302,3600)),
                        ds['lat'].values.reshape((1,2302,3600)))
    ct = gsw.CT_from_pt(sa, pt)
    dens = gsw.rho(sa, ct, p)
    # 保存结果到新的 netcdf 文件中
    ds_density = xr.Dataset({
        'density': (['lev1','y','x'], dens),
        'lon':     (['y','x'], ds['lon'].values),
        'lat':     (['y','x'], ds['lat'].values),
        'lev1':    (['lev1'], ds['lev1'].values)
    })
    # 利用文件名最后部分作为区分
    out_file = f'{dens_path}/density-{filename[-16:-3]}.nc'
    ds_density.to_netcdf(out_file)
    print(f"Rank {rank} saved {out_file}")