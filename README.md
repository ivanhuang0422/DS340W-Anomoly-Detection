# DRAEM & DRL

PyTorch implementation of [DRAEM](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf) - ICCV2021:

```
@InProceedings{Zavrtanik_2021_ICCV,
    author    = {Zavrtanik, Vitjan and Kristan, Matej and Skocaj, Danijel},
    title     = {DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {8330-8339}
}
```
## Datasets
To train on the MVtec Anomaly Detection dataset [download](https://www.mvtec.com/company/research/datasets/mvtec-ad)
the data and extract it. The [Describable Textures dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) was used as the anomaly source 
image set in most of the experiments in the paper. You can run the **download_dataset.sh** script from the project directory
to download the MVTec and the DTD datasets to the **datasets** folder in the project directory:
```
./scripts/download_dataset.sh
```

## Subsampling
The **create_mvtec_subsample.py** script maintains the full structure of the MVTec AD dataset while creating a smaller subset. The script requires the path to the original MVTec AD dataset (--source), where you want the subsampled dataset to be created (--target), fractions of image to keep (default: 0.2 = 20%), and random seed for reproducibility (--seed). The subsample script can be run with:

```
python create_mvtec_subsample.py --source ./mvtec_anomaly_detection --target ./mvtec_subset --ratio 0.2
```

## Training
Pass the folder containing the training dataset to the **train.py** script as the --data_path argument and the
folder locating the anomaly source images as the --anomaly_source_path argument. 
The training script also requires the batch size (--bs), learning rate (--lr), epochs (--epochs), path to store checkpoints
(--checkpoint_path) and path to store logs (--log_path).
Example:

```
python train.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/
```

Train on subsampled data example:
```
python train.py --gpu_id 0 --obj_id -1 --lr 0.001 --bs 8 --epochs 5  --data_path ./datasets/mvtec_subset/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/ 
```

The conda environement used in the project is decsribed in **requirements.txt**.

## Pretrained models
Pretrained DRAEM models for each class of the MVTec anomaly detection dataset are available [here](https://drive.google.com/uc?id=1eOE8wXNihjsiDvDANHFbg_mQkLesDrs1).
To download the pretrained models directly see **./scripts/download_pretrained.sh**.

The pretrained models achieve a 98.1 image-level ROC AUC, 97.5 pixel-wise ROC AUC and a 68.9 pixel-wise AP.


## Evaluating
The test script requires the --gpu_id arguments, the name of the checkpoint files (--base_model_name) for trained models, the 
location of the MVTec anomaly detection dataset (--data_path) and the folder where the checkpoint files are located (--checkpoint_path)
with pretrained models can be run with:

```
python test.py --gpu_id 0 --base_model_name "DRAEM_test_0.001_5_bs8" --data_path ./datasets/mvtec/ --checkpoint_path ./checkpoints/DRAEM_checkpoints/
```



