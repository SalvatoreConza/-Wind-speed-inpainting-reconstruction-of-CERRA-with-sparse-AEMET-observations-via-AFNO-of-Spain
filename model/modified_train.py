# # MAIN MODIFICATIONS: 
# - modification in the function train_on_epoch adding the mask
# - modification in the funcation validate_on_epoch adding the mask
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Iterable, Sequence, Type, Union
import warnings

import torch
from torch import Tensor

import os
import xarray as xr
from datetime import datetime
import numpy as np

# import FuseAdam if available for optimized training
try:
    from apex.optimizers import FusedAdam
except ImportError:
    warnings.warn("Apex is not installed, defaulting to PyTorch optimizers.")

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger
# from physicsnemo.launch.utils import load_checkpoint #, save_checkpoint
from model.modified_checkpoint import load_checkpoint, save_checkpoint
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
import glob
import re


class Trainer:
    """Training loop for diagnostic models."""

    def __init__(
        self,
        model: Module,
        dist_manager: DistributedManager,
        loss: Callable,
        train_datapipe: Sequence,
        valid_datapipe: Sequence,
        input_output_from_batch_data: Union[Callable, None] = None,
        optimizer: Union[Type[torch.optim.Optimizer], None] = None,
        optimizer_params: Union[dict, None] = None,
        scheduler: Union[Type[torch.optim.lr_scheduler.LRScheduler], None] = None,
        scheduler_params: Union[dict, None] = None,
        max_epoch: int = 1,
        patience: int = 50,
        save_best_checkpoint: bool = True,
        load_epoch: Union[int, str, None] = None,
        inference_on_epoch: Union[str, int] = "best",
        checkpoint_every: int = 1,
        checkpoint_dir: Union[str, None] = None,
        validation_callbacks: Iterable[Callable] = (),
    ):
        self.model = model # The neural network model (a physicsnemo.Module) that will be trained.
        self.dist_manager = dist_manager # A helper object to manage distributed training
        self.loss = loss # : The loss function (e.g., Mean Squared Error) that measures how "wrong" the model's predictions are.
        self.train_datapipe = train_datapipe # data loader
        self.valid_datapipe = valid_datapipe # data loader
        self.max_epoch = max_epoch # The maximum number of times to loop through the entire training dataset.
        if input_output_from_batch_data is None:
            input_output_from_batch_data = lambda x: x
        self.input_output_from_batch_data = input_output_from_batch_data
        # creation of the optimizer and the learning rate scheduler by calling the setup functions
        self.optimizer = self._setup_optimizer(
            opt_cls=optimizer, opt_params=optimizer_params
        )
        self.lr_scheduler = self._setup_lr_scheduler(
            scheduler_cls=scheduler, scheduler_params=scheduler_params
        )
        self.validation_callbacks = list(validation_callbacks)
        self.inference_on_epoch = inference_on_epoch
        self.device = self.dist_manager.device
        self.logger = PythonLogger()

        # setup the initial state of the trainer from a checkpoint if provided
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.epoch = 1 # set the starting epoch to 1
        # se load_epoch è fornito immediatamente carica pesi e numero di epoche da un checkpoint
        if load_epoch is not None:
            epoch = None if load_epoch == "latest" else load_epoch
            self.load_checkpoint(epoch=epoch)

        # Early stopping, tecnique to automatically stop the trainining if the model isn't improving
        self.save_best_checkpoint = save_best_checkpoint
        self.best_val_loss = float("inf")  # store the best validation loss observed so far, starting from infinity
        self.best_epoch = 0
        self.best_model_state_dict = None
        self.patience = patience           # numbers oe epoch the trainer will wait for the validation loss to improve before giving up
        self.bad_epochs = 0                # counter for how many epoch have passed without improvement


        # wrap capture here instead of using decorator so it'll still be wrapped if
        # overridden by a subclass. this wrappers add more logics to train_step_forward and eval_step:

        # train_step_forward return the loss, the loss is taken by loss.backward to calculate the gradient
        # and then we update the model's weights with optimizer.step()
        self.train_step_forward = StaticCaptureTraining(
            model=self.model,
            optim=self.optimizer,
            logger=self.logger,
            use_graphs=False,  # for some reason use_graphs=True causes a crash
        )(self.train_step_forward)
        
        self.eval_step = StaticCaptureEvaluateNoGrad(
            model=self.model, logger=self.logger, use_graphs=False
        )(self.eval_step)

    def eval_step(self, invar: Tensor) -> Tensor:
        """Perform one step of model evaluation."""
        # return the raw prediction of the inputs components (invar[0], invar[1])
        return self.model(invar)

    def train_step_forward(self, invar: Tensor, outvar_true: Tensor, mask: Tensor) -> Tensor:
        """Train model on one batch."""
        # takes the inputs data invar, passess its two components invar[0] and
        # invar[1] to the model and generate model's prediction outvar_pred
        outvar_pred = self.model(invar)  # invar[0] is the input, invar[1] is the ratio
        # calculates the loss between the model's prediction and the ground truth outvar_true and return it
        return self.loss(outvar_pred, outvar_true,mask)
    
    # main training driver
    def fit(self):
        """Main function for training loop."""
        # main loop over epochs, go from self.epoch (1 or loaded from checkpoint) to self.max_epoch and 
        # a each ereration calls self.train_on_epoch to perform training for one epoch
        for self.epoch in range(self.epoch, self.max_epoch + 1):
            self.train_on_epoch() 

            # Early stopping check, after one epoch is done it checks if the validation loss has failed to improve for self.patience
            if self.bad_epochs >= self.patience:
                print(f"Early stopping triggered at epoch {self.epoch}")
                break
        
        if self.dist_manager.rank == 0:
            self.logger.info("Finished training!")

            # Save best model weights using PhysicsNemo logic
            if self.save_best_checkpoint and self.best_model_state_dict is not None:
                self.model.load_state_dict(self.best_model_state_dict)
                self.epoch = self.best_epoch
                self.save_checkpoint(base_name="best")
                print(
                    f"Saved best model from epoch {self.best_epoch} "
                    f"with val_loss={self.best_val_loss:.5f}"
                )

    def train_on_epoch(self):
        """Train for one epoch."""
        # initialize logging for track the training progress
        with LaunchLogger(
            "train",
            epoch=self.epoch,
            num_mini_batch=len(self.train_datapipe),
            epoch_alert_freq=10,
        ) as log:
            # loop over training batches from the self.train_datapipe 
            for batch in self.train_datapipe:
                inputs, target, mask, *_ = self.input_output_from_batch_data(batch)    
                
                # perform one training step and get the loss
                loss = self.train_step_forward(
                    inputs, target, mask)
                log.log_minibatch({"loss": loss.detach()})

            log.log_epoch({"Learning Rate": self.optimizer.param_groups[0]["lr"]})

        # run Validation, get the validation val_loss and compares to the self.best_val_loss
        if self.dist_manager.rank == 0:
            with LaunchLogger("valid", epoch=self.epoch) as log:
                val_loss = self.validate_on_epoch()
                log.log_epoch({"Validation error": val_loss})

            # check if this is the best model. if this epoch is better update the best_val_loss:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = self.epoch

                # and store the best model weight in memory:
                self.best_model_state_dict = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
                self.bad_epochs = 0  # reset patience
            else:
                self.bad_epochs += 1 # increment patience counter is this is not the best model

        # sync all processes, it makes all processes wait here until the slowest one reach this point (useful in training on multiple GPU)
        if self.dist_manager.world_size > 1:
            torch.distributed.barrier()

        # updtate the learning rate for the next epoch
        self.lr_scheduler.step()

        # save a periodic checkpoint, this is for resuming training in case of crash
        checkpoint_epoch = (self.checkpoint_dir is not None) and (
            (self.epoch % self.checkpoint_every == 0) or (self.epoch == self.max_epoch)
        )
        if checkpoint_epoch and self.dist_manager.rank == 0:
            # Save PhysicsNeMo Launch checkpoint
            self.save_checkpoint(base_name="checkpoint")

    @torch.no_grad() # disable gradient calculation for validation, you dont need to perform gradient calculations when evaluating the model
    # validate_on_epoch is resposnbile for evaluating the model on the validation dataset ater each training epoch
    def validate_on_epoch(
        self,
        perform_inference: bool = False, 
        save_dir: str = "./results/maps",
        prediction_var_name: str = "predicted",
        mean = None,
        std = None,
    ) -> Tensor:
        """Return average loss over one validation epoch."""
        loss_epoch = 0
        num_examples = 0  # Number of validation examples
        # Dealing with DDP wrapper
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        preds = []
        times = []
        # this section prepares the means and std tensors for reverse normalization during inference
        if mean is not None and std is not None:
            mean = torch.from_numpy(mean).to(device=self.device)
            std = torch.from_numpy(std).to(device=self.device)
            # reverse normalization
            if mean.ndim == 1:  # (C,)
                mean = mean.view(1, -1, 1, 1)
                std = std.view(1, -1, 1, 1)
            elif mean.ndim == 3:  # (C, H, W)
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected shape for mean: {mean.shape}")
        # main part where the model is evalueted    
        try:
            model.eval() # we switch the model to evaluation mode, disabling dropout and batchnorm updates that are active only during training
            # loop thourgh every batch
            for (i, batch) in enumerate(self.valid_datapipe):
                #(invar, outvar_true, coords_dict, mask) = self.input_output_from_batch_data(batch)
                #invar = (invar.detach())
                #outvar_true = outvar_true.detach()
                
                # --- 
                invar, outvar_true, coords_dict, mask = batch
                # Move tensors to GPU (Manually, to skip the dict)
                invar = invar.to(self.device)
                outvar_true = outvar_true.to(self.device)
                mask = mask.to(self.device)
                # 
                #  ---
                

                outvar_pred = self.eval_step(invar) # forward pass to get model predictions
                loss_epoch += self.loss(outvar_pred, outvar_true, mask) # it calculates the loss for this batch and adds it to the total loss for the epoch
                num_examples += 1

                for callback in self.validation_callbacks:
                    callback(outvar_true, outvar_pred, epoch=self.epoch, batch_idx=i)
                if perform_inference:
                    if mean is not None and std is not None:
                        # reverse normalization
                        outvar_pred_unnorm = outvar_pred * std + mean
 
                    preds.append(outvar_pred_unnorm.cpu())
                    times.extend([np.datetime64(t) for t in coords_dict["time"]])
        finally:  # restore train state so that when the next training epoch starts dropout/batchnorm will be enabled again
            model.train()

        # saving predictions 
        if perform_inference and self.dist_manager.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            
            preds_np = torch.cat(preds, dim=0).numpy() # torch.cat take the list of batch predictions and concatenate them into 
            # a single large tensor along the time dimension
            times_np = np.array(times, dtype="datetime64[ns]")

            # Since coords_dict is drawn from the data loader, it comes in batches, e.g. coords_dict["lat"] has shape (batchs_size, lat)
            lat = coords_dict["lat"][0]
            lon = coords_dict["lon"][0]

            coords = {
                "time": times_np,
                "latitude": lat,  # static
                "longitude": lon,
            }

            # Only add "channel" if not splitting per-variable
            if not isinstance(prediction_var_name, list):
                coords["channel"] = np.arange(preds_np.shape[1])

            # If multiple variable names are provided, split channels and assign each
            if isinstance(prediction_var_name, list):
                if len(prediction_var_name) != preds_np.shape[1]:
                    raise ValueError(f"Number of variable names ({len(prediction_var_name)}) "
                                     f"does not match number of channels ({preds_np.shape[1]}) in predictions.")

                data_vars = {
                    var_name: (["time", "latitude", "longitude"], preds_np[:, i])
                    for i, var_name in enumerate(prediction_var_name)
                }
            else:
                # Fallback: use a single variable name
                data_vars = {
                    prediction_var_name: (["time", "channel", "latitude", "longitude"], preds_np)
                }

            ds = xr.Dataset(data_vars, coords=coords) # create an xarray Dataset to store the predictions along with their coordinates

            if isinstance(prediction_var_name, list):
                var_str = "_".join(prediction_var_name)
            else:
                var_str = prediction_var_name

            timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            netcdf_path = os.path.join(save_dir, f"{var_str}_{timestamp}.nc")
            ds.to_netcdf(netcdf_path)
            print(f"✅ Saved predictions to NetCDF: {netcdf_path}")

        return loss_epoch / num_examples # return the average validation loss (ie the toal loss divided by the number of batches)

    # this function intializes the optimizer which is the algorithm used to update the model's weights:
    def _setup_optimizer(self, opt_cls=None, opt_params=None):
        """Initialize optimizer."""
        opt_kwargs = {"lr": 0.0005} # default learning rate
        # if you provide your own optimizer parameters 
        if opt_params is not None:
            opt_kwargs.update(opt_params)
        # if you don't provide your own optimizer class it first try to use FusedAdam if available otherwise default to AdamW
        if opt_cls is None:
            try:
                opt_cls = FusedAdam
            except NameError:  # in case we don't have apex
                opt_cls = torch.optim.AdamW
        # creates and returns the optimizer instance, telling it which parameters to train:
        return opt_cls(self.model.parameters(), **opt_kwargs)

    # this function initializes the learning rate scheduler which adjusts the learning rate during training
    def _setup_lr_scheduler(self, scheduler_cls=None, scheduler_params=None):
        """Initialize learning rate scheduler."""
        scheduler_kwargs = {}
        # case if no scheduler_cls class is provided:
        if scheduler_cls is None:
            scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
            scheduler_kwargs["T_max"] = self.max_epoch # This tells the scheduler to complete one full "annealing" 
            # (so from high LR to low LR) over the total number of training epochs.
        # case if scheduler_cls class is provided:
        if scheduler_params is not None:
            scheduler_kwargs.update(scheduler_params)

        return scheduler_cls(self.optimizer, **scheduler_kwargs) #return the created scheduler 
    
    # function for loading a checkpoint to resume training from a saved state
    def load_checkpoint(self, epoch: Union[int, None] = None) -> int:
        """Load training state from checkpoint.

        Parameters
        ----------
        epoch: int or None, optional
            The epoch for which the state is loaded. If None, will load the
            latest epoch.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in order to load checkpoints.")
        # updates self.epoch to the saved epoch number:
        self.epoch = load_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            device=self.device,
            epoch=epoch,
        )
        return self.epoch
    # function for saving the complete training state, saves model weights (self.model), 
    # optimizer state( self.optimizer), scheduler state(self.lr_scheduler) and current epoch number (self.epoch)
    def save_checkpoint(self, base_name: str = "checkpoint"):
        """Save training state from checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set in order to save checkpoints.")
        save_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            epoch=self.epoch,
            base_name=base_name
        )
        
    # function for loading ONLY the model weights for inference after training is complete
    def load_model_for_inference(self):
        """Loads the best model saved with base_name='best'."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set to load checkpoints.")
        if self.inference_on_epoch == "best":
            # Obtain epoch corresponding to best
            # Find all checkpoint files with 'best' prefix
            best_checkpoints = glob.glob(f"{self.checkpoint_dir}/best.*.pt")
            if not best_checkpoints:
                self.logger.warning("No best checkpoint found. Loading latest checkpoint instead.")
                epoch = None
            else:
                # Extract epoch number from filenames
                epochs = []
                for ckpt in best_checkpoints:
                    match = re.search(r'best\.\d+\.(\d+)\.pt', ckpt)
                    if match:
                        epochs.append(int(match.group(1)))
                
                if epochs:
                    epoch = max(epochs)  # Get the highest epoch number
                    self.logger.info(f"Loading best model from epoch {epoch}")
                else:
                    epoch = None
                    self.logger.warning("Could not parse epoch from best checkpoint filename. Loading latest.")
            load_checkpoint(
                path=self.checkpoint_dir,
                models=self.model,
                optimizer=None,
                scheduler=None,
                epoch=epoch,  # or None if you just want latest
                device=self.device,
                base_name="best"
            )
        else:
            load_checkpoint(
                path=self.checkpoint_dir,
                models=self.model,
                optimizer=None,
                scheduler=None,
                epoch=self.inference_on_epoch,
                device=self.device,
            )

        


