# pixelNeRF-YOLO: Multi-view object detection using NeRF and YOLO

Nándor Kőfaragó<br>
Budapest University of Technology and Economics

Paper available here: https://tdk.bme.hu/VIK/DownloadPaper/Tobbnezetu-objektumdetektalas-NeRF-es-YOLO

## Environment setup

The easiest to start is by creating a conda environment.

```sh
conda env create -f environment.yml
conda activate pixelnerf
```

Or by installing the packages from `requirements.txt`.

## Getting the data

Download the dataset here: https://drive.google.com/file/d/1ilFjfpR_pe5z4oOpGJEZ4Fz47UNZVBWu/view?usp=drive_link

## Training the model

To train the model with the encoder, run this in repository root folder:

```sh
python train/train.py -n yolo --dataset_format yolo -c conf/exp/yolo.conf -D ./data/yolo -V 3 --gpu_id=0 --resume -B 1 --gamma 0.9 --epochs 50
```

To train without training the encoder run:

```sh
python train/train.py -n yolo --dataset_format yolo -c conf/exp/yolo.conf -D ./data/yolo -V 3 --gpu_id=0 --resume -B 1 --gamma 0.9 --epochs 50 --freeze_enc
```

All the settings can be found in `conf/exp/yolo.conf`, use backbone `custom` or `resnet34` to use YOLO or ResNet encoder.

To use YOLO encoder, clone https://github.com/szemenyeim/NeRF-YOLO to the same folder as this repository and install the pretrained weights (from: https://github.com/WongKinYiu/yolov7).

## Running evaluation

To run the evaluation, use:

```sh
python eval/eval_yolo.py -n yolo --dataset_format yolo -c conf/exp/yolo.conf -D ./data/yolo -V 3 --gpu_id=0 --resume -B 1
```

## Running visualization

To run a visualization, use:

```sh
python eval/gen_images_yolo.py -n yolo --dataset_format yolo -c conf/exp/yolo.conf -D ./data/yolo -V 3 --gpu_id=0 -B 1 --source "0 2 5" --dest 0 --resume
```

Specify the input and target views with `--source` and `--dest`.