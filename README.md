# LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents

This repository contains the code for [LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents](https://arxiv.org/abs/2104.00249) by ByeoungDo Kim, Seong Hyeon Park, Seokhwan Lee, Elbek Khoshimjonov, Dongsuk Kum, Junsoo Kim, Jeong Soo Kim, Jun Won Choi

## Dataset Preprocessing
```sh
python dataset_preprocess.py -p [dataset-path]
```
## Model Training - nuScenes
```sh
python run.py -m Lapred_original
```
## Model Evaluation - nuScenes
```sh
python run.py -m Lapred_original -e
```
