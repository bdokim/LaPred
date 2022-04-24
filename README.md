# LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents

This repository contains the code for [LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents](https://arxiv.org/abs/2104.00249) by ByeoungDo Kim, Seong Hyeon Park, Seokhwan Lee, Elbek Khoshimjonov, Dongsuk Kum, Junsoo Kim, Jeong Soo Kim, Jun Won Choi

## Dataset
- Download [nuScenes dataset](https://www.nuscenes.org/nuscenes#download)

## Dataset Preprocessing
- Run the script to extract preprocessed samples.
- Provide the path of the downloaded data to --path(-p) option. (default : './nuscenes/dataset')
```sh
python dataset_preprocess.py -p [dataset-path]
```
## Model Training - nuScenes
- To train the LaPred model, run the run.py file.
```sh
python run.py -m Lapred_original
```
## Model Evaluation - nuScenes
- After training, You can evaluate the model with --eval(-e) option.
```sh
python run.py -m Lapred_original -e
```
