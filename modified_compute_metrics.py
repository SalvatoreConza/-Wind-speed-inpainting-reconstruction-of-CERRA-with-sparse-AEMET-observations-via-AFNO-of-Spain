# inference_results/maps/ws_2025_11_28_09:11:38.nc
import os
import numpy as np #library for numerical computations
import xarray as xr #library for handling multidimensional arrays 
import torch # deep learning library
import hydra # configuation tool, helps manage settings via yaml files
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim # popular metric for comparing 2 images or 2D arrays

# hydra main decorator to specify config file locationm loading and create config object:
@hydra.main(
    version_base=None, config_path="config", config_name="modified_config_file.yaml"
)
# convert config yaml file to standard python dict and pass it to compute_metrics function:
def main(cfg):
    compute_metrics(**OmegaConf.to_container(cfg)) 

def compute_metrics(**cfg):
    # Load dataset, read the path fro m cfg dictionary and use xr.open_dataset to load prediction and ground truth datasets:
    pred_ds = xr.open_dataset(os.path.join(cfg["metrics"]["pred_path"], cfg["metrics"]["pred_dataset"]))
    gt_ds = xr.open_dataset(os.path.join(cfg["metrics"]["gt_path"], cfg["metrics"]["gt_dataset"]))
    
    if 'latitude' in pred_ds.dims and 'y' in gt_ds.dims:
        print("Renaming dimensions: 'latitude' -> 'y', 'longitude' -> 'x'")
        pred_ds = pred_ds.rename({'latitude': 'y', 'longitude': 'x'})
        
    # Also check the reverse (just in case)
    elif 'y' in pred_ds.dims and 'latitude' in gt_ds.dims:
         pred_ds = pred_ds.rename({'y': 'latitude', 'x': 'longitude'})
    
    # Ensure the datasets have the same dimensions, safety check:
    assert pred_ds.dims == gt_ds.dims, "Prediction and ground truth datasets must have the same dimensions."

    # Compute metrics for each variable via a loop, first he check if the variable is present in both dataset 
    # then calls compute_axis_aware_metrics function to calculate metrics and then stores them inthe metric dictionary
    metrics = {}
    for var in cfg["metrics"]["variables"]:
        if var in pred_ds and var in gt_ds:
            metrics[var] = compute_axis_aware_metrics(pred_ds[var], gt_ds[var])
        else:
            print(f"Variable {var} not found in both datasets.")

    # loop over the computed metrics and write them to a text file:
    output_file = cfg["metrics"]["metrics_file"]
    with open(output_file, 'w') as f:
        for var, metric in metrics.items():
            f.write(f"Metrics for {var}:\n")
            for key, value in metric.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

# this funcition takes the preditction and ground truth data and compute a wide range of metrics 
# along the axes/dimensions time, latitude, longitude
def compute_axis_aware_metrics(pred: xr.DataArray, gt: xr.DataArray):
    assert pred.shape == gt.shape, "Prediction and target must match in shape." #sanity check

    # Convert to torch tensors the raw numerical data from xarray
    pred = torch.from_numpy(pred.values).float()
    gt   = torch.from_numpy(gt.values).float()

    # # Mask NaNs
    # mask = (~np.isnan(pred)) & (~np.isnan(gt))
    # pred = pred.where(mask)
    # gt = gt.where(mask)

    # --- RMSE, MSE & MAE: overall. This metrics are calculated by plattening the entier dataset into a single list of numbers ---
    rmse_total = ((pred - gt) ** 2).mean().sqrt() # The square root of MSE
    mse_total = ((pred - gt) ** 2).mean() # The average of the squared differences
    mae_total = (pred - gt).abs().mean() # The average of the absolute differences

    # --- RMSE/MAE/MSE over time axis (per pixel, then averaged) ---
    # .mean(dim=0): Averages along the time dimension
    # .mean(): Averages this 2D error map into a single number.
    rmse_over_time = ((pred - gt) ** 2).mean(dim=0).sqrt().mean()
    mse_over_time = ((pred - gt) ** 2).mean(dim=0).mean()
    mae_over_time = (pred - gt).abs().mean(dim=0).mean() 

    # --- RMSE/MAE/MSE over space axis (per timestep, then averaged) ---
    # .mean(dim=[1, 2]): Averages along the spatial dimensions.
    # .mean(): Averages this 1D error time series into a single number.
    rmse_over_space = ((pred - gt) ** 2).mean(dim=[1, 2]).sqrt().mean()
    mse_over_space = ((pred - gt) ** 2).mean(dim=[1, 2]).mean()
    mae_over_space = (pred - gt).abs().mean(dim=[1, 2]).mean()
    
    # This measures how well the patterns of the prediction match the patterns of the ground truth:
    # --- Pearson Correlation over time (for each spatial point, then averaged) ---
    def time_corr(pred, gt): # Calculates the correlation for each pixel across the time axis.
        x_mean = pred.mean(dim=0)
        y_mean = gt.mean(dim=0)
        cov = ((pred - x_mean) * (gt - y_mean)).mean(dim=0)
        std_x = pred.std(dim=0)
        std_y = gt.std(dim=0)
        return (cov / (std_x * std_y)).mean()  # mean over space

    # --- Pearson Correlation over space (for each time step, then averaged) ---
    def spatial_corr(pred, gt): # Calculates the correlation for each time step across the spatial dimensions.
      x_mean = pred.mean(dim=[1, 2])
      y_mean = gt.mean(dim=[1, 2])
      cov = ((pred - x_mean[:, None, None]) * (gt - y_mean[:, None, None])).mean(dim=[1, 2])
      std_x = pred.std(dim=[1, 2])
      std_y = gt.std(dim=[1, 2])
      return (cov / (std_x * std_y)).mean()

    corr_time = time_corr(pred, gt)
    corr_space = spatial_corr(pred, gt)

    # --- R² Score: Overall Variance Explained ---
    # 1.0 is a perfect score (model explains all variance).
    # 0.0 means the model is no better than just predicting the average value every time.
    ss_res = ((gt - pred) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)

    # --- SSIM: Spatial fidelity (time averaged), is a maetric that compares two 
    # images, considering structure, contrast, and luminance. It's more sophisticated than RMSE. ---
    ssim_vals = []
    # loop through each time step, calculates SSIM between the predicted and ground truth 2D arrays and than averages all the SSIM scores:
    for t in range(pred.shape[0]):
        x = pred[t].cpu().numpy()
        y = gt[t].cpu().numpy()
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            continue
        score = ssim(x, y, data_range=y.max() - y.min())
        ssim_vals.append(score)
    ssim_mean = np.mean(ssim_vals) if ssim_vals else np.nan

    # --- PSNR: Peak Signal-to-Noise Ratio ---
    # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE), so it compares the maximum possible value of data to the amount of error
    # Higher values indicate better quality
    if mse_total.item() > 0:
        max_pixel_value = max(gt.max().item(), pred.max().item())
        psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse_total.item())
    else:
        psnr = float('inf')  # Perfect match case


    # --- Physical Constraint: Total Mass Error (spatial sum) ---
    # sums all pixel values for each time step (.sum(dim=[1, 2])) to get the total mass.
    total_mass_pred = pred.sum(dim=[1, 2]) 
    total_mass_gt = gt.sum(dim=[1, 2])
    # average absolute error between the predicted total mass and the ground truth total mass over time.
    mass_error = (total_mass_pred - total_mass_gt).abs().mean() 

    # gather and bundles all the metrics calculated into a single dictionary 
    return {
        "RMSE_total": rmse_total.item(),
        "MAE_total": mae_total.item(),
        "MSE_total": mse_total.item(),
        "RMSE_over_time": rmse_over_time.item(),
        "MAE_over_time": mae_over_time.item(),
        "MSE_over_time": mse_over_time.item(),
        "RMSE_over_space": rmse_over_space.item(),
        "MAE_over_space": mae_over_space.item(),
        "MSE_over_space": mse_over_space.item(),
        "Corr_over_time": corr_time.item(),
        "Corr_over_space": corr_space.item(),
        "R_squared": r_squared.item(),
        "SSIM_mean": ssim_mean,
        "PSNR": psnr,
        "Mass_Conservation_Error": mass_error.item()
    }


if __name__ == "__main__":
    main()
