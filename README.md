# Wind Speed Inpainting: CERRA Reconstruction via AFNO

This repository contains an implementation for reconstructing high-resolution wind speed fields (CERRA reanalysis) from sparse observational data (AEMET stations) using an Adaptive Fourier Neural Operator (AFNO). The model performs **inpainting**, effectively filling in the missing spatial data based on sparse inputs to approximate the complete CERRA reanalysis field.

## 📂 Project Structure

The repository is organized into modular components for configuration, data loading, model definition, and execution.

### 🚀 Key Execution Scripts
These are the primary entry points for running the workflow:

* **`modified_main_training.py`**: The main driver for model training. It parses the configuration file, loads the data, builds the model, and initializes the `Trainer` class to execute the training loop. First training is performed here, followed by evaluation using the separate evaluation script.
* **`modified_eval.py`**: The evaluation script. It loads the best model checkpoint saved during training, runs inference on a dedicated test dataset, and generates the final model predictions.
* **`modified_compute_metrics.py`**: A post-processing script that takes the model's predictions and the ground truth data to calculate key performance metrics (e.g., RMSE, MAE) to quantify reconstruction accuracy.

### ⚙️ Configuration
* **`config/modified_config_file.yaml`**: The central control file for the entire project. It manages parameters for both training and evaluation, defining:
    * **Model Architecture:** Hyperparameters for the AFNO model.
    * **Data Paths:** Locations of the CERRA and AEMET NetCDF files.
    * **Hyperparameters:** Settings for learning rate, batch size, epochs, etc.
    * **Classes:** Configuration sections corresponding to model, sources, training, evaluation, inference, args, logging, and metrics.

### 🧠 Model & Utilities (`model/`)
* **`model/modified_loss.py`**: Defines the optimization objectives (loss functions) as classes used to minimize the error between reconstructed wind fields and ground truth during training.
* **`model/modified_train.py`**: Contains the generic `Trainer` class. This handles the core logic: iterating over epochs and batches, performing forward and backward passes, running validation loops, implementing early stopping, and managing checkpoints.
* **`model/modified_normalization.py`**: A helper module for `modified_dataloader.py`. It calculates necessary statistics (Mean & Std or Min & Max) required for the normalization strategies defined in the dataset class.
* **`model/modified_checkpoint.py`**: A utility used by the `Trainer` class to handle disk I/O for checkpoints. It manages saving checkpoints, loading them, and determining the best checkpoint based on validation performance.

### 💾 Data Pipeline (`data/`)
* **`data/modified_dataloader.py`**: Defines the `NetCDF_Dataset` class tailored for the inpainting task. It includes:
    * Four specific normalization functions.
    * Logic to ingest raw CERRA and AEMET data.
    * Organization of dynamic variables.
    * Output generation of tensors ready for the neural network.

---

## 📉 Loss Functions

The training objective combines two specific loss components to ensure the model reconstructs both the global field and the specific valid data points accurately.

### 1. Whole Loss ($L_{whole}$)
This measures the reconstruction error over the entire spatial domain (all grid points), encouraging the model to generate a physically consistent global wind field.

$$L_{whole} = || M \odot (Y_{pred} - Y_{gt}) ||_1$$

Where:
* $M$ is a binary mask (typically all 1s for the whole domain, or weighting valid regions).
* $Y_{pred}$ is the predicted wind field.
* $Y_{gt}$ is the ground truth (CERRA).
* $\odot$ denotes the Hadamard (element-wise) product.

### 2. Hole Loss ($L_{hole}$)
This focuses exclusively on the "holes" (missing data regions) or the valid station points, depending on the masking strategy. It forces the model to be extremely accurate at the specific locations where ground truth observations exist.

$$L_{hole} = || (1 - M) \odot (Y_{pred} - Y_{gt}) ||_1$$

The final optimization objective is a weighted sum of these two terms.

---

## 🔬 Key Modifications

This implementation includes two significant architectural and preprocessing enhancements over standard AFNO approaches:

### 1. Nearest Neighbor Interpolation for Stations
To handle the sparsity of the AEMET station data effectively, we apply **Nearest Neighbor Interpolation** during the preprocessing stage. Instead of feeding the model a mostly empty grid with single-pixel values at station locations, this technique projects station values onto the nearest grid points of the CERRA mesh. This provides a denser and more representative initial input map for the network, reducing the difficulty of the inpainting task.

### 2. Partial Convolution in Patch Embedding
We modified the **Patch Embedding** layer in `afno.py` to use **Partial Convolutions** (PConv). Standard convolution treats valid pixels and masked zeros equally, which can lead to artifacts where the "missing" values corrupt the features at the boundaries of valid data. 

Partial Convolution updates the mask as it processes the image, re-normalizing the convolution output based only on the valid pixels in the sliding window. This ensures that the initial feature extraction step is robust to the high sparsity of the input data, passing cleaner features to the subsequent Fourier Neural Operator blocks.

---

## 🛠️ Usage

### 1. Configure the Experiment
Before running any scripts, edit the configuration file to match your environment and experiment needs.
* **Target File:** `config/modified_config_file.yaml`
* **Action:** Update data paths, adjust AFNO architecture parameters, or modify training hyperparameters (learning rate, epochs, etc.).

### 2. Train the Model
To start the training process:
```bash
python modified_main_training.py
```
Wind speed inpainting from AEMET station for CERRA reanalysis reconstruction via AFNO. 
- Input variable: windspeed 
- Patch size: 10x10, embedded dimension 512, depth 8, number of blocks 4, dropout=0.0
- Trained for 171 epoch with patience of 40 epocs
<img width="2700" height="750" alt="comparison_plot_fixed (1)" src="https://github.com/user-attachments/assets/c5327112-98f7-4b0c-bcd3-50d72b00d784" />
<img width="1919" height="1079" alt="Screenshot 2025-12-03 220911" src="https://github.com/user-attachments/assets/cadc2ad7-382d-4906-a1d1-d0f5362290ba" />
<img width="1919" height="1033" alt="Screenshot 2025-12-03 220859" src="https://github.com/user-attachments/assets/ac8ba313-ceaa-4502-b338-8807cf30adaf" />
<br />
<br />
<br />

- Input variable: windspeed
- Patch size: 10x10, embedded dimention 768, depth 12, number of blocks 8, dropout=0.2
- Trained for 201 epoch with patience of 40 epocs
<img width="2700" height="750" alt="comparison_plot_fixed" src="https://github.com/user-attachments/assets/e6ae712e-609f-4a7e-ab22-4b918502a361" />

You can see how the model learn better the global patterns that bound very far pixel so he predict better the windspeed in the ocean. Since the model is much more 
complex I have introduced I preventivly introduced a dropout for reducting overfitting but this seems by the metrics to constrain the model to not learning enough
<br />
<br />
<br />

- Input variable: windspeed
- Patch size: 10x10, embedded dimention 768, depth 12, number of blocks 8, dropout=0.0
- Trained for 201 epoch with patience of 40 epocs
<br />
<br />
<br />
