# LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents

This repository contains the code for [LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents](https://arxiv.org/abs/2104.00249) by ByeoungDo Kim, Seong Hyeon Park, Seokhwan Lee, Elbek Khoshimjonov, Dongsuk Kum, Junsoo Kim, Jeong Soo Kim, Jun Won Choi

## Dataset Preprocessing

python dataset_preprocess.py

## Model Training - nuScenes

python train.py -m Lapred_original

## Model Evaluation - nuScenes

python train.py -m Lapred_original -e
