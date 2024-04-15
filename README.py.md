# small_gicp (fast_gicp2)

**small_gicp** is a header-only C++ library that offers efficient and parallelized algorithms for fine point cloud registration (ICP, Point-to-Plane ICP, GICP, VGICP, etc.). It is a refined and optimized version of its predecessor, [fast_gicp](https://github.com/SMRT-AIST/fast_gicp), re-written from scratch with the following features.

- **Highly Optimized** : The implementation of the core registration algorithm is further optimized from that in fast_gicp. It enables up to **2x speed gain** compared to fast_gicp.
- **All parallerized** : small_gicp offers parallelized implementations of several preprocessing algorithms to make the entire registration process parallelized (Downsampling, KdTree construction, Normal/covariance estimation). As a parallelism backend, either (or both) [OpenMP](https://www.openmp.org/) and [Intel TBB](https://github.com/oneapi-src/oneTBB) can be used. 
- **Minimum dependency** : Only [Eigen](https://eigen.tuxfamily.org/) (and bundled [nanoflann](https://github.com/jlblancoc/nanoflann) and [Sophus](https://github.com/strasdat/Sophus)) are required at a minimum. Optionally, it provides the [PCL](https://pointclouds.org/) registration interface so that it can be used as a drop-in replacement in many systems.
- **Customizable** : small_gicp allows feeding any custom point cloud class to the registration algorithm via traits. Furthermore, the template-based implementation enables customizing the registration process with your original correspondence estimator and registration factors.
- **Python bindings** : The isolation from PCL makes small_gicp's python bindings more portable and connectable to other libraries (e.g., Open3D) without problems. 

Note that GPU-based implementations are NOT included in this package.

If you find this package useful for your project, please consider leaving a comment [here](https://github.com/koide3/small_gicp/issues/3). It would help the author receive recognition in his organization and keep working on this project.


[![Build](https://github.com/koide3/small_gicp/actions/workflows/build.yml/badge.svg)](https://github.com/koide3/small_gicp/actions/workflows/build.yml) [![Test](https://github.com/koide3/small_gicp/actions/workflows/test.yml/badge.svg)](https://github.com/koide3/small_gicp/actions/workflows/test.yml) [![codecov](https://codecov.io/gh/koide3/small_gicp/graph/badge.svg?token=PCVIUP2Z33)](https://codecov.io/gh/koide3/small_gicp)

## Installation

### Python

**Install from PyPI**

```bash
pip install small_gicp --user
```

or **install from source**

```bash
git clone https://github.com/koide3/small_gicp
cd small_gicp
pip install . --user
```

## Usage (Python) [basic_registration.py](src/example/basic_registration.py)

Example A : Perform registration with numpy arrays

```python
# Arguments
# - target_points               : Nx4 or Nx3 numpy array of the target point cloud
# - source_points               : Nx4 or Nx3 numpy array of the source point cloud
# Optional arguments
# - init_T_target_source        : Initial guess of the transformation matrix (4x4 numpy array)
# - registration_type           : Registration type ("ICP", "PLANE_ICP", "GICP", "VGICP")
# - voxel_resolution            : Voxel resolution for VGICP
# - downsampling_resolution     : Downsampling resolution
# - max_correspondence_distance : Maximum correspondence distance
# - num_threads                 : Number of threads
result = small_gicp.align_points(target_raw_numpy, source_raw_numpy, downsampling_resolution=0.25)

result.T_target_source  # Estimated transformation (4x4 numpy array)
result.converged        # If true, the optimization converged successfully
result.iterations       # Number of iterations the optimization took
result.num_inliers      # Number of inlier points
result.H                # Final Hessian matrix (6x6 matrix)
result.b                # Final information vector (6D vector)
result.e                # Final error (float)
```

Example B : Perform preprocessing and registration separately

```python
# Preprocess point clouds
# Arguments
# - points_numpy                : Nx4 or Nx3 numpy array of the target point cloud
# Optional arguments
# - downsampling_resolution     : Downsampling resolution
# - num_neighbors               : Number of neighbors for normal and covariance estimation
# - num_threads                 : Number of threads
target, target_tree = small_gicp.preprocess_points(points_numpy=target_raw_numpy, downsampling_resolution=0.25)
source, source_tree = small_gicp.preprocess_points(points_numpy=source_raw_numpy, downsampling_resolution=0.25)

# `target` and `source` are small_gicp.PointCloud with the following methods
target.size()           # Number of points
target.points()         # Nx4 numpy array   [x, y, z, 1] x N
target.normals()        # Nx4 numpy array   [nx, ny, nz, 0] x N
target.covs()           # Array of 4x4 covariance matrices

# Align point clouds
# Arguments
# - target                      : Target point cloud (small_gicp.PointCloud)
# - source                      : Source point cloud (small_gicp.PointCloud)
# - target_tree                 : KD-tree of the target point cloud (small_gicp.KdTree)
# Optional arguments
# - init_T_target_source        : Initial guess of the transformation matrix (4x4 numpy array)
# - max_correspondence_distance : Maximum correspondence distance
# - num_threads                 : Number of threads
result = small_gicp.align(target, source, target_tree)
```

Example C : Perform each of preprocessing steps one-by-one

```python
# Convert numpy arrays (Nx3 or Nx4) to small_gicp.PointCloud
target_raw = small_gicp.PointCloud(target_raw_numpy)
source_raw = small_gicp.PointCloud(source_raw_numpy)

# Downsampling
target = small_gicp.voxelgrid_sampling(target_raw, 0.25)
source = small_gicp.voxelgrid_sampling(source_raw, 0.25)

# KdTree construction
target_tree = small_gicp.KdTree(target)
source_tree = small_gicp.KdTree(source)

# Estimate covariances
small_gicp.estimate_covariances(target, target_tree)
small_gicp.estimate_covariances(source, source_tree)

# Align point clouds
result = small_gicp.align(target, source, target_tree)
```

Example D: Example with Open3D

```python
target_o3d = open3d.io.read_point_cloud('small_gicp/data/target.ply').paint_uniform_color([0, 1, 0])
source_o3d = open3d.io.read_point_cloud('small_gicp/data/source.ply').paint_uniform_color([0, 0, 1])

target, target_tree = small_gicp.preprocess_points(points_numpy=numpy.asarray(target_o3d.points), downsampling_resolution=0.25)
source, source_tree = small_gicp.preprocess_points(points_numpy=numpy.asarray(source_o3d.points), downsampling_resolution=0.25)
result = small_gicp.align(target, source, target_tree)

source_o3d.transform(result.T_target_source)
open3d.visualization.draw_geometries([target_o3d, source_o3d])
```

## License
This package is released under the MIT license.

If you find this package useful for your project, please consider leaving a comment [here](https://github.com/koide3/small_gicp/issues/3). It would help the author receive recognition in his organization and keep working on this project.

## Contact

[Kenji Koide](https://staff.aist.go.jp/k.koide/), National Institute of Advanced Industrial Science and Technology (AIST)
