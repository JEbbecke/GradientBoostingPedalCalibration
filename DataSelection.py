import numpy as np

# %% load linear calibrated pilot data to identify cycling specific force values (AOI = Area Of Interest)
# pilot_data mus be n x 3 numpy array
pilot_data = np.load(r"pathto\PilotData.npy")


# %% load training data
# training_data mus be n x 6 numpy with columns 1-3 force inputs and 4-6 sensor outputs
training_data = np.load(r"pathto\TrainingData.npy")


# %% Define calibration space and divide it into voxels of specified size
# Define range of interest for each axis
x_range = (-300, 300)
y_range = (-300, 300)
z_range = (-200, 600)

# Define the number of voxels along each axis
num_voxels_x = 30
num_voxels_y = 30
num_voxels_z = 40

# Calculate the size of each voxel along each axis
voxel_size_x = (x_range[1] - x_range[0]) / num_voxels_x
voxel_size_y = (y_range[1] - y_range[0]) / num_voxels_y
voxel_size_z = (z_range[1] - z_range[0]) / num_voxels_z

# Calculate voxel indices for each point in all_data to determine which voxel it belongs to AOI
voxel_indices_x = np.floor((pilot_data[:, 0] - x_range[0]) / voxel_size_x).astype(int)
voxel_indices_y = np.floor((pilot_data[:, 1] - y_range[0]) / voxel_size_y).astype(int)
voxel_indices_z = np.floor((pilot_data[:, 2] - z_range[0]) / voxel_size_z).astype(int)

# Filter out points outside the defined ranges
valid_points = (voxel_indices_x >= 0) & (voxel_indices_x < num_voxels_x) & \
               (voxel_indices_y >= 0) & (voxel_indices_y < num_voxels_y) & \
               (voxel_indices_z >= 0) & (voxel_indices_z < num_voxels_z)

# Apply the filter
voxel_indices_x = voxel_indices_x[valid_points]
voxel_indices_y = voxel_indices_y[valid_points]
voxel_indices_z = voxel_indices_z[valid_points]

# Identify unique voxels that contain at least one data point. Ideally, all voxels have a minimum number of 10 data points
voxel_idx = np.unique(np.vstack((voxel_indices_x, voxel_indices_y, voxel_indices_z)).T, axis=0)


# %% select data points
selected_data = []

# Convert voxel_idx to a set of tuples for efficient lookup
aoi_voxels_set = set(tuple(v) for v in voxel_idx)

# Pre-compute the voxel indices for training_data
training_voxel_indices = np.floor((training_data[:, [0, 1, 2]] - np.array([x_range[0], y_range[0], z_range[0]])) / 
                                  np.array([voxel_size_x, voxel_size_y, voxel_size_z])).astype(int)

# Iterate over all possible voxels
o = 1
for i in range(num_voxels_x):
    for j in range(num_voxels_y):
        for k in range(num_voxels_z):
            print(f'Processing voxel {o} of {num_voxels_x * num_voxels_y * num_voxels_z}')
            o += 1
            # Check if the current voxel is in the AOI
            is_in_aoi = (i, j, k) in aoi_voxels_set

            # Determine the number of points to select from this voxel
            num_points_per_voxel_selected = 10 if is_in_aoi else 5

            # Create a mask to filter training data points within this voxel
            mask = (training_voxel_indices[:, 0] == i) & \
                   (training_voxel_indices[:, 1] == j) & \
                   (training_voxel_indices[:, 2] == k)

            # Find indices of training_data that are within the current voxel
            indices_within_voxel = np.where(mask)[0]

            # Select random data points from this voxel
            if indices_within_voxel.size > 0:
                selected_indices = np.random.choice(indices_within_voxel, size=min(num_points_per_voxel_selected, indices_within_voxel.size), replace=False)
                selected_data.extend(training_data[selected_indices])

# Convert selected data to numpy array
selected_data = np.array(selected_data)


# %% check data selection quality
# check density count datapoints in selected data per voxel
num_points_per_voxel_selected = np.zeros((num_voxels_x, num_voxels_y, num_voxels_z))
num_points_per_voxel_training = np.zeros((num_voxels_x, num_voxels_y, num_voxels_z))
o = 1
for i in range(num_voxels_x):
    for j in range(num_voxels_y):
        for k in range(num_voxels_z):
            print(f'Processing voxel {o} of {num_voxels_x * num_voxels_y * num_voxels_z}')
            o += 1
            x_min = x_range[0] + i * voxel_size_x
            x_max = x_min + voxel_size_x
            y_min = y_range[0] + j * voxel_size_y
            y_max = y_min + voxel_size_y
            z_min = z_range[0] + k * voxel_size_z
            z_max = z_min + voxel_size_z

            num_points_per_voxel_training[i, j, k] = np.sum((training_data[:,0] >= x_min) & (training_data[:,0] < x_max) & \
                                                    (training_data[:,1] >= y_min) & (training_data[:,1] < y_max) & \
                                                    (training_data[:,2] >= z_min) & (training_data[:,2] < z_max))

            num_points_per_voxel_selected[i, j, k] = np.sum((selected_data[:,0] >= x_min) & (selected_data[:,0] < x_max) & \
                                                    (selected_data[:,1] >= y_min) & (selected_data[:,1] < y_max) & \
                                                    (selected_data[:,2] >= z_min) & (selected_data[:,2] < z_max))
            

# give stats on data selection
num_5 = np.sum(num_points_per_voxel_selected == 5)
num_20 = np.sum(num_points_per_voxel_selected == 10)
num_0 = np.sum(num_points_per_voxel_selected == 0)
num_any = np.sum(num_points_per_voxel_selected > 0)

print(f'Total number of voxels: {num_voxels_x * num_voxels_y * num_voxels_z}')
print(f'Number of voxels in AOI: {voxel_idx.shape[0]} ({voxel_idx.shape[0] / (num_voxels_x * num_voxels_y * num_voxels_z) * 100:.2f}% of total voxels)')
print(f'Number of empty voxels: {num_0} ({num_0 / (num_voxels_x * num_voxels_y * num_voxels_z) * 100:.2f}% of total voxels)')
print(f'Number of voxels with 5 points: {num_5} ({num_5 / ((num_voxels_x * num_voxels_y * num_voxels_z) - voxel_idx.shape[0]) * 100:.2f}% of non-AOI voxels)')
print(f'Number of voxels with 20 points: {num_20} ({num_20 / (voxel_idx.shape[0]) * 100:.2f}% of AOI voxels)')
print(f'Number of voxels with any points: {num_any} ({num_any / (num_voxels_x * num_voxels_y * num_voxels_z) * 100:.2f}% of total voxels)')