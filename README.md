Files to run:
- project/modified_main_training.py
- project/modified_eval.py
- project/modified_compute_metrics.py

File to modify for the parameters/files on which do the inference:
- project/config/modified_config_file.yaml


Wind speed inpainting from AEMET station for CERRA reanalysis reconstruction via AFNO. 
- Input variable: windspeed 
- Patch size: 10x10, embedded dimension 512, depth 8, number of blocks 4
- Trained for 171 epoch with patience of 40 epocs
<img width="2700" height="750" alt="comparison_plot_fixed (1)" src="https://github.com/user-attachments/assets/c5327112-98f7-4b0c-bcd3-50d72b00d784" />
<img width="1919" height="1079" alt="Screenshot 2025-12-03 220911" src="https://github.com/user-attachments/assets/cadc2ad7-382d-4906-a1d1-d0f5362290ba" />
<img width="1919" height="1033" alt="Screenshot 2025-12-03 220859" src="https://github.com/user-attachments/assets/ac8ba313-ceaa-4502-b338-8807cf30adaf" />

- Input variable: windspeed 
- Patch size: 10x10, embedded dimention 768, depth 12, number of blocks 8
- Trained for 201 epoch with patience of 40 epocs
<img width="2700" height="750" alt="comparison_plot_fixed" src="https://github.com/user-attachments/assets/e6ae712e-609f-4a7e-ab22-4b918502a361" />

You can see how the model learn better the global patterns that bound very far pixel so he predict better the windspeed in the ocean
