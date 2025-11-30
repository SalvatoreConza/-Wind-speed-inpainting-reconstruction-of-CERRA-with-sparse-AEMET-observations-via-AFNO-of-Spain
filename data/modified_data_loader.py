# MAIN MODIFICATIONS: 
# - modifing the class NetCDFDataset
# - modifing the final ouput
# IGNORE FOLLOWING COMMENTS:

import torch
from torch.utils.data import Dataset
import xarray as xr

# definining 4 helper normalization functions

# channel wise normalization using mean and std
def ZScore(x, mean, std):
    # --- SAFETY CHECK START ---
    # If the statistics are missing (None), we cannot normalize.
    # We return the raw 'x' to prevent the crash.
    if mean is None or std is None:
        return x
    # --- SAFETY CHECK END ---

    # Original logic:
    # We use 1e-5 for eps to prevent division by zero
    return (x - mean[:, None, None]) / (std[:, None, None] + 1e-5)


# pixel wise normalization using mean and std
def ZScorePerPixel(x, mean, std, eps=1e-8):
    """
    Apply pixel-wise z-score normalization.
    x: torch.Tensor of shape [C, H, W]
    mean, std: torch.Tensor of shape [C, H, W]
    """
    
    return (x - mean) / (std + eps)

# channel wise normalization using min max
def MinMax(x, min_val, max_val, eps=1e-8):
    """
    Apply pixel-wise min-max normalization.
    x: torch.Tensor of shape [C, H, W]
    min_val, max_val: torch.Tensor of shape [C,]
    """
    return (x - min_val[:, None, None]) / (max_val[:, None, None] - min_val[:, None, None] + eps)

# pixel wise normalization using min max
def MinMaxPerPixel(x, min_val, max_val, eps=1e-8):
    """
    Apply pixel-wise min-max normalization.
    x: torch.Tensor of shape [C, H, W]
    min_val, max_val: torch.Tensor of shape [C, H, W]
    """
    return (x - min_val) / (max_val - min_val + eps)
# eps is a small number addedd to the denominator to avoid division by zero.

# Custom PyTorch Dataset class
class NetCDFDataset(Dataset):
    # It accept three xarray datasets objects:
    # input_static_ds: static data that doesn't change over time like elevation
    # input_ds_1: low-res dynamic data that change over time like low res pm2.5
    # target_ds: high-res dynamic data that change over time like high res pm2.5

    # static_variables, input_variables, output_variables are the variables that we want
    # so I filter the three xarray datasets 

    def __init__(self, input_ds_1: xr.Dataset, target_ds: xr.Dataset, mask_ds: xr.Dataset = None,
                 input_variables: list = ['ws'], output_variables: list = ['ws'], mask_variables: list = ['ws'],  
                 input_stats = None, target_stats = None, norm_method="none"):
        # storing the datasets and parameters as internal variables
        self.input_ds_1 = input_ds_1
        self.target_ds = target_ds
        # Validate shape compatibility (debugginG)
        ### The following is for training, in inference it may be a problem if we do not have target data. 
        # assert len(input_ds_1.time) == len(target_ds.time), "Time dimensions must match"

        # stroring the lists of variables we care about
        self.input_vars = input_variables
        self.output_vars = output_variables

        self.mask_var = mask_variables
        # --- 1. MASK LOADING FIX ---
        self.mask_data = None
        if mask_ds is not None:
            # Check for variables availability
            available_mask_vars = [v for v in self.mask_var if v in mask_ds.data_vars]
            if available_mask_vars:
                # Convert Dataset -> DataArray -> Numpy
                # We transpose to (time, variable, y, x) to match input structure
                # This fixes the "TypeError: method" by ensuring we get values from a DataArray
                mask_array = mask_ds[available_mask_vars].to_array().transpose("time", "variable", "y", "x")
                self.mask_data = torch.tensor(mask_array.values, dtype=torch.float32)
                print(f"Mask loaded successfully. Shape: {self.mask_data.shape}")
            else:
                print("Warning: Requested mask variables not found in mask_ds.")
                
        # filter variables, check if the variables requested are presente in the 3 xarray datasets provided
        available_input_vars = [v for v in self.input_vars if v in input_ds_1.data_vars]
        available_output_vars = [v for v in self.output_vars if v in target_ds.data_vars]

        # in the three successive line of code we converst the data from xarray datasets to array and then to torch tensors
        # to_array() stacks the different variables into a single new dimeension
        # .transpose(...)  reordering 
        self.input_data = input_ds_1[available_input_vars].to_array().transpose("time", "variable", "y", "x")
        self.target_data = target_ds[available_output_vars].to_array().transpose("time", "variable", "y", "x")    #.sel(level=0.0).squeeze("level")  # Drop level dim if needed
        
        # storing time, lat, lon (i.e. the coordinates) for later use
        self.times = target_ds.time.values
        self.lats = target_ds.y.values
        self.lons = target_ds.x.values

        # preparation to Normalization, so we save mean,std,min,max, depending on the if
        # conditions fullfilled, so depending on the normalization method requested
        self.norm_method = norm_method
        if norm_method == "z_score" or norm_method == "z_score_per_pixel":
            if input_stats is not None:
                self.input_mean = torch.tensor(input_stats['mean'], dtype=torch.float32)
                self.input_std = torch.tensor(input_stats['std'], dtype=torch.float32)
            else:
                self.input_mean = self.input_std = None 
            if target_stats is not None:
                self.target_mean = torch.tensor(target_stats['mean'], dtype=torch.float32)
                self.target_std = torch.tensor(target_stats['std'], dtype=torch.float32)
            else:
                self.target_mean = self.target_std = None
        elif norm_method == "minmax" or norm_method == "minmax_per_pixel":
            if input_stats is not None:
                self.input_min = torch.tensor(input_stats['min'], dtype=torch.float32)
                self.input_max = torch.tensor(input_stats['max'], dtype=torch.float32)
            if target_stats is not None:
                self.target_min = torch.tensor(target_stats['min'], dtype=torch.float32)
                self.target_max = torch.tensor(target_stats['max'], dtype=torch.float32)
            else:
                self.target_min = self.target_max = None
        elif norm_method == "none":
            self.norm_method = "none"
            self.input_mean = self.input_std = None
            self.target_mean = self.target_std = None
            self.input_min = self.input_max = None
            self.target_min = self.target_max = None
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")

    # return the length of the dataset
    def __len__(self):
        return len(self.input_ds_1.time)

    # get one single item from my dataset, process it and return it
    def __getitem__(self, idx):
        # [idx] slice one single time
        # .values converts Xarray slice into Numpy array
        # toch.tensort() convets numpy array into torch tensor
        # Lazy load tensors using dask -> numpy -> torch
        # x1, x2 and y are the tensors for dynamic input (low res pm), static input (elevation) and target (high res pm) respectively
        x = torch.tensor(self.input_data[idx].values, dtype=torch.float32)  # (variables, lat, lon)
        y = torch.tensor(self.target_data[idx].values, dtype=torch.float32)  # (lat', lon')
        # x2 = torch.full((1,), 1/self.high_res_low_res_ratio, dtype=torch.float32)  # (1,)
        # actual Normalization
        if self.norm_method == "z_score":
            if self.input_mean is not None and self.input_std is not None:
                x = ZScore(x, self.input_mean, self.input_std)
                y = ZScore(y, self.target_mean, self.target_std)

        elif self.norm_method == "z_score_per_pixel":
            if self.input_mean is not None and self.input_std is not None:
                x = ZScorePerPixel(x, self.input_mean, self.input_std)
                y = ZScorePerPixel(y, self.target_mean, self.target_std)

        elif self.norm_method == "minmax":
            if self.input_min is not None and self.input_max is not None:
                x = MinMax(x, self.input_min, self.input_max)
                y = MinMax(y, self.target_min, self.target_max)

        elif self.norm_method == "minmax_per_pixel":
            if self.input_min is not None and self.input_max is not None:
                x = MinMaxPerPixel(x, self.input_min, self.input_max)
                y = MinMaxPerPixel(y, self.target_min, self.target_max)

        elif self.norm_method == "none":
            print("No normalization applied to input data.")
        
        # LOAD MASK (Updated)
        if self.mask_data is not None:
            # mask_data is numpy array [Time, H, W]
            mask = self.mask_data[idx]             
            # IMPORTANT: Add Channel Dimension
            # Current shape: [250, 250] -> Needed: [1, 250, 250]
            mask = mask.unsqueeze(0)
        else:
            # Fallback if no mask provided (returns all zeros or ones depending on logic)
            # Assuming 0=Valid, 1=Hole
            h, w = y.shape[-2], y.shape[-1]
            mask = torch.zeros((1, h, w), dtype=torch.float32)
            
        # creation of a python dictionary to hold the labels
        # High-res coords for this specific sample
        coords = {
            "time": str(self.times[idx]),    # scalar
            "lat": self.lats,                # (lat',)
            "lon": self.lons                 # (lon',)
        }

        # final ouput of the function: tuple containing the dynnamical and static data, target tensor and metadata dictionary
        return x, y, coords, mask