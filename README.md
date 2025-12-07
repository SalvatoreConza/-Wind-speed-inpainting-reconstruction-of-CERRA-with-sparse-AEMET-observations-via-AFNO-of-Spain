Files to run:
- modified_main_training.py
- modified_eval.py
- modified_compute_metrics.py

File to modify for the parameters/files on which do the inference:
- config/modified_config_file.yaml

modiefied_config_file.yaml controls both training and evaluation. It fedines the model architecture. data paths and hyperparameters. 
It's made my the classes model, sources, training, evaluation, inference, training, args, logging and metrics.

modified_main_training.py is the primary script for training the model. First I do training here and then the evaluation running another file. 
It read the the yaml file, load the datas, create the model and initialize the trainer class.

modified_dataloader.py have 4 functions for the normalization and define a class NetCFD_Dataset that prepare the datas for an inpainting problem. 
It takes the datas of CERRA and AEMET as input, it organizes them taking dynamical variables, normalize them and give as output some tensors.

modified_normalization.py is a helper to the modified_dataloader.py. 
After specified what normalizzation use in the NetCFD_Dataset this file help us to calculat mean and std (or min & max)

modified_loss.py defines the goal of training, so the loss that we want minimize. It contains different losses defined as classes.

modified_train.py contains the generic train class, in which we have the iterations over epoch, batches, performing formward and backward passes, running validation, 
implementing early stoppung and saving checkpoints.

modified_checkpoint,py is an utility, used by the trainer class defined in modified_train.py, to save and load the checkpoints to and from the disk and also how to 
chose the best checkpoint

modified_eval.py is a script that take the best checkpoint, load it, run it on a test dataset and produce the model's final prediction

modified_compute_metrics.py is a script that take the ground truth and the prediction and evaluate some key metrics like RMSE and MAE.

Wind speed inpainting from AEMET station for CERRA reanalysis reconstruction via AFNO. 
- Input variable: windspeed 
- Patch size: 10x10, embedded dimension 512, depth 8, number of blocks 4
- Trained for 171 epoch with patience of 40 epocs
<img width="2700" height="750" alt="comparison_plot_fixed (1)" src="https://github.com/user-attachments/assets/c5327112-98f7-4b0c-bcd3-50d72b00d784" />
<img width="1919" height="1079" alt="Screenshot 2025-12-03 220911" src="https://github.com/user-attachments/assets/cadc2ad7-382d-4906-a1d1-d0f5362290ba" />
<img width="1919" height="1033" alt="Screenshot 2025-12-03 220859" src="https://github.com/user-attachments/assets/ac8ba313-ceaa-4502-b338-8807cf30adaf" />
<br />
<br />
<br />

- Input variable: windspeed
- Patch size: 10x10, embedded dimention 768, depth 12, number of blocks 8
- Trained for 201 epoch with patience of 40 epocs
<img width="2700" height="750" alt="comparison_plot_fixed" src="https://github.com/user-attachments/assets/e6ae712e-609f-4a7e-ab22-4b918502a361" />

You can see how the model learn better the global patterns that bound very far pixel so he predict better the windspeed in the ocean
