# Vegetation Forecasting

## Download

- **Pretrained checkpoint**  
  [Google Drive – checkpoint folder](https://drive.google.com/drive/folders/1V4JXgLtGWFQXE4SF7qN5N4vn6-b58kgZ?usp=sharing)

- **Sample data**  
  [Google Drive – data_sample folder](https://drive.google.com/drive/folders/1YKF5ce5w3SfkaVZftg_fDCzIwo88B_2J?usp=sharing)
  
## Conda environment
**environment.yaml**
## Training

```bash
python train.py seed=42.yaml --data_dir /path/to/data_sample


## Usage

1. **Edit the checkpoint path**  
   In `my_contextformer.py`, make sure you point `--clip_checkpoint` (and any other `checkpoint` args) at the folder or file where you’ve stored your pretrained weights. Example:  
   ```python
   # inside my_contextformer.py
   CLIP_CHECKPOINT = "/path/to/your/checkpoint-169410"
