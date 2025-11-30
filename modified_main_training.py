# MAIN MODIFICATIONS: 
# - remove the reading of static variables,
# - modifing the initialization of the model from modAFNO to AFNO
# - modification of the loading loop
# IGNORE FOLLOWING COMMENTS:
import os
import sys
# torch, torch.nn, torch.optim are core deep learning imports:
import torch
# xarray is for handling netCDF data:
import xarray as xr
# hydra e omegaconfig are for managing learning rates, model size, file path via an external yaml file
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import numpy as np
import torch.nn as nn
from torchinfo import summary
# modafno is the specific NN importedi from a local model directory
from model.afno import AFNO

# physicsnemo is a custom library for MLFflow and distributed training
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.distributed.manager import DistributedManager

from model import modified_train
from data.modified_data_loader import NetCDFDataset
from model.modified_normalization import compute_stats, save_stats
from model.modified_loss import GeneralPM25Loss
import torch.optim as optim
# dataloader is for batching and loading data:
from torch.utils.data import DataLoader

# load experiment settings from hydra config file:
@hydra.main(
    version_base=None, config_path="config", config_name="modified_config_file.yaml"
)

# convert of config from omegaconfig to standard python dict and 
# pass all the setting from the yaml file to the training function:
def main(cfg):
    train_downscaling(**OmegaConf.to_container(cfg))

# helper funciton to log experiment setting to a MLflow server
def log_hyperparameters(cfg, client, run):
    # this function converst a nested dict like cfg into a flat dict, so  
    # as an example {'model': {'lr': 0.01}} becomes {'model.lr': 0.01}
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_cfg = flatten_dict(cfg)
    # loops over every single hyperparameter and logs it to MLflow>
    for key, value in flat_cfg.items():
        try:
            client.log_param(run.info.run_id, key, value)
        except Exception as e:
            print(f"Could not log param {key}: {e}")

# simple function to count the number of trainable parameters (i.e the one requiring gradients) in the model and sum up their total elements:
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_downscaling(**cfg):
    # environment setup, seed for reproduciability, use the gpu if available otherwise cpu
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # calculate number of static channels:
    static_schannels = 0
    # Model setup, initializate the modAFNO, passing all the relevant hyperparameters from 
    # the config file to the model constructior and moves the model onto the GPU if avaiable
    model = AFNO(
        inp_shape=cfg["model"]["inp_shape"],
        #out_shape=cfg["model"]["out_shape"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        # embed_model=cfg["model"]["embed_model"],
        patch_size=cfg["model"]["patch_size"],
        embed_dim=cfg["model"]["embed_dim"],
        # mod_dim=cfg["model"]["embed_model"]["dim"],
        depth=cfg["model"]["depth"],
        num_blocks=cfg["model"]["num_blocks"],
        # modulate_mlp=cfg["model"]["modulate_mlp"],
        # modulate_filter=cfg["model"]["modulate_filter"],
    ).to(device)

    # count_parameters print the toal number of trainable parameters in the model
    # summary print a detailed summary of the model architecture
    print(f"Number of trainable parameters: {count_parameters(model):,}")
    summary(
        model, 
        input_size=(
            1,                              # Batch size
            cfg["model"]["in_channels"],    # Total Channels (Variables * Time steps)
            cfg["model"]["inp_shape"][0],   # Height (y)
            cfg["model"]["inp_shape"][1]    # Width (x)
        ),            
        device=device.type
    )
    # Dataset setup
    # Static high-res features
    # Reads all .nc files in the directory of the static features and merges them into a single xarray Dataset
    # static_files = sorted([os.path.join(cfg["sources"]["dataset"]["static_features_path"], f) for f in os.listdir(cfg["sources"]["dataset"]["static_features_path"]) if f.endswith(".nc")])
    # input_static_ds = xr.open_mfdataset(static_files,  combine='by_coords', join='outer')



    # Input low-res features
    train_input_ds = xr.open_dataset(cfg["sources"]["dataset"]["train_input"])
    # input mask
    train_mask_ds = xr.open_dataset(cfg["sources"]["dataset"]["train_mask"])
    # Target high-res features
    train_target_ds = xr.open_dataset(cfg["sources"]["dataset"]["train_target"])

    # Normalization 
    # Compute and save normalization independent of whether stats exist or not
    norm_cfg = cfg['sources']["dataset"]["normalization"]
    # compute & persist input train‐set stats
    input_stats = compute_stats(train_input_ds,
                          cfg["sources"]["dataset"]["input_variables"],
                          norm_cfg["method"])
    os.makedirs(os.path.dirname(norm_cfg["input_stats_path"]) or ".", exist_ok=True)
    save_stats(input_stats, norm_cfg["input_stats_path"])
    print(f"→ saved normalization stats ({norm_cfg['method']}) to {norm_cfg['input_stats_path']}")

    if norm_cfg['compute_target_stats']:
        target_stats = compute_stats(train_target_ds,
                                      cfg["sources"]["dataset"]["output_variables"],
                                      norm_cfg["method"])
        save_stats(target_stats, norm_cfg["target_stats_path"])


    train_dataset = NetCDFDataset(
        train_input_ds,    # Positional Arg 1 (Likely maps to input_dataset)
        train_target_ds,   # Positional Arg 2 (Likely maps to target_dataset)
        train_mask_ds,
        # --- DELETED THE LINE BELOW ---
        # cfg["sources"]["dataset"], 
        # ------------------------------
        
        # Explicit Keyword Arguments
        input_variables=cfg["sources"]["dataset"]['input_variables'],
        output_variables=cfg["sources"]["dataset"]['output_variables'],
        mask_variables=cfg["sources"]["dataset"]['mask_variables'],
        input_stats=input_stats,
        target_stats=target_stats,
        norm_method=norm_cfg["method"],
    
        # Since you removed the 'ratio' positional argument from the original, 
        # you might need to pass it as a keyword if your class expects it:
        # high_res_low_res_ratio=cfg["sources"]["dataset"].get('high_res_low_res_ratio', 1) 
    )
    
    valid_input_ds = xr.open_dataset(cfg["sources"]["dataset"]["val_input"])
    valid_target_ds = xr.open_dataset(cfg["sources"]["dataset"]["val_target"])
    valid_mask_ds = xr.open_dataset(cfg["sources"]["dataset"]["val_mask"])

    valid_dataset = NetCDFDataset(
        valid_input_ds,    # Positional Arg 1 (Likely maps to input_dataset)
        valid_target_ds,   # Positional Arg 2 (Likely maps to target_dataset)
        valid_mask_ds,
        # --- DELETED THE LINE BELOW ---
        # cfg["sources"]["dataset"], 
        # ------------------------------
        # Explicit Keyword Arguments
        input_variables=cfg["sources"]["dataset"]['input_variables'],
        output_variables=cfg["sources"]["dataset"]['output_variables'],
        mask_variables=cfg["sources"]["dataset"]['mask_variables'],
        input_stats=input_stats,
        target_stats=target_stats,
        norm_method=norm_cfg["method"],
    
        # Since you removed the 'ratio' positional argument from the original, 
        # you might need to pass it as a keyword if your class expects it:
        # high_res_low_res_ratio=cfg["sources"]["dataset"].get('high_res_low_res_ratio', 1) 
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['training_args']['batch_size'], shuffle=True, num_workers=cfg['training_args']['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['training_args']['batch_size'], shuffle=False, num_workers=cfg['training_args']['num_workers'])

    # Distributed setup
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    print(f"Using device: {dist_manager.device}")

    # Logging
    mlflow_cfg = cfg.get("logging", {}).get("mlflow", {}).copy()

    if mlflow_cfg.pop("use_mlflow", False):
        # Manually resolve any ${now:...} expressions in run_name
        run_name = mlflow_cfg.get("run_name", "")
        if "${now:" in run_name:
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            mlflow_cfg["run_name"] = run_name.replace("${now:%Y-%m-%d_%H-%M-%S}", now_str)

        client, run = initialize_mlflow(**mlflow_cfg)
        LaunchLogger.initialize(use_mlflow=True)

        log_hyperparameters(cfg, client, run)

    # Loss
    loss_func = GeneralPM25Loss(
        loss_type=cfg["training_args"]["loss"]["loss_type"],
        log_weight=cfg["training_args"]["loss"]["log_weight"],
        eps=cfg["training_args"]["loss"]["eps"]
    ).to(device)
 
    print('loss:', loss_func)

    # Optimizer
    optimizer_cls = getattr(optim, cfg["training_args"]["optimizer"]["optimizer_type"])
    optimizer_params = cfg["training_args"]["optimizer"].get("optimizer_params", {})

    # setup training loop
    def input_output_from_batch_data(batch):
    # 1. Input Image -> GPU
        inputs = batch[0].to(device)
    
        # 2. Target Image -> GPU
        # Since your dataloader is fixed, this is now a Tensor!
        # We can send it to the GPU directly.
        targets = batch[1].to(device)

        # 3. Metadata (Keep as is)
        metadata = batch[2]
        # 4 mask
        mask = batch[3].to(device)

        return inputs, targets, mask, metadata
 
    
    trainer = modified_train.Trainer(
        model,
        dist_manager=dist_manager,
        loss=loss_func,
        train_datapipe=train_loader,
        valid_datapipe=valid_loader,
        
        # USE THE FINAL FUNCTION:
        input_output_from_batch_data=input_output_from_batch_data,
        
        optimizer=optimizer_cls,
        optimizer_params=optimizer_params,
        **cfg["training"],
    )

    # train model
    trainer.fit()


if __name__ == "__main__":
    main()

