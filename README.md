# AHATrack


## Prepare Environment

Ensure that you have install the `pytorch` >= 2.2 and corresponding `torchvision` version.  It doesn't matter whether you use pip or conda.

Then execute
```sh
bash install.sh
```
You can check the script where I install the `opencv-python-headless` library because of my headless server environment.  If you are running in a machine with GUI, you can comment that out.

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Training and Evaluation on Datasets
We support training following [OSTrack](https://github.com/botaoye/OSTrack).
### Dataset Preparation
Put the tracking datasets in ./data. It should look like this:
```
${PROJECT_ROOT}
 -- data
     -- lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- got10k
         |-- test
         |-- train
         |-- val
     -- coco
         |-- annotations
         |-- images
     -- trackingnet
         |-- TRAIN_0
         |-- TRAIN_1
         ...
         |-- TRAIN_11
         |-- TEST
``` 
### Training
Download pre-trained [CAE ViT-Base weights]()(cae_base.pth) and put it under  `$PROJECT_ROOT$/pretrained_models`.   
> **NOTE**: ViT in CAE is slightly different from the original ones (in Image is worth 16x16 words and MAE), e.g., projections of Q,K,V and layer scale.  Details can be seen in the code.

Run the command below to train the model:
```sh
# single GPU
python tracking/train.py --script ahatrack --config B8_cae_center_got10k_ep100 --save_dir ./output --mode single  --use_wandb 0

# 2 GPUs
python tracking/train.py --script ahatrack --config B8_cae_center_got10k_ep100 --save_dir ./output --mode multiple --nproc_per_node 2  --use_wandb 0
```
- If you want to use wandb to record detailed log and get clear curve of training, set `--use_wandb 1`.  
- The batch size, learning rate and the other detailed parameters can be set in config file, e.g., `experiments/ahatrack/B8_cae_center_got10k_ep100.yaml`.
- In our implemention, we always follow `batch_size` $\times$ `num_GPU` $\div$ `learning_rate` $= 32 \times 1 \div 0.0001$ for alignment with OSTrack.  For example, if u use 4 GPUs and bs=32 (in each GPU), then the lr should be 0.0004. 
- We save the checkpoints without optimization params in the last 20% epochs, for testing over these epochs to avoid accuracy jittering.

### Evaluation
Use your own training weights or ours in `$PROJECT_ROOT$/output/checkpoints/train/ahatrack`.  
Some testing examples:

- LaSOT
```sh
python tracking/test.py ahatrack B8_cae_center_all_ep300 --dataset lasot --threads 8 --num_gpus 1 --ep 300 299 290
python tracking/analysis_results.py # need to modify tracker configs and names
```
For other off-line evaluated benchmarks, modify --dataset correspondingly.

- GOT10K-test 
```sh
python tracking/test.py ahatrack B8_cae_center_got10k_ep100 --dataset got10k_test --threads 8 --num_gpus 1 --ep 100 99 98
python lib/test/utils/transform_got10k.py --tracker_name atatrack --cfg_name B8_cae_center_got10k_ep100_099 # the last number is epoch
```
- TrackingNet (cost ~1 hour on 2080Ti with i5-11400F for one epoch testing)
```sh
python tracking/test.py ahatrack B8_cae_center_all_ep300 --dataset trackingnet--threads 8 --num_gpus 1 --ep 300 299
python lib/test/utils/transform_trackingnet.py --tracker_name atatrack --cfg_name B8_cae_center_all_ep300_300 # the last number is epoch
```
