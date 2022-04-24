# LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents

This repository contains the code for [LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents](https://arxiv.org/abs/2104.00249) by ByeoungDo Kim, Seong Hyeon Park, Seokhwan Lee, Elbek Khoshimjonov, Dongsuk Kum, Junsoo Kim, Jeong Soo Kim, Jun Won Choi

![lapred_img](https://user-images.githubusercontent.com/16588420/164984523-dceea537-eeb3-43a8-b7c3-0144bd9e5f50.jpg)

```bibtex
@InProceedings{Kim_2021_CVPR,
    author    = {Kim, ByeoungDo and Park, Seong Hyeon and Lee, Seokhwan and Khoshimjonov, Elbek and Kum, Dongsuk and Kim, Junsoo and Kim, Jeong Soo and Choi, Jun Won},
    title     = {LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {14636-14645}
}
```

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
