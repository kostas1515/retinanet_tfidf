# pytorch-retinanet
Forked and modified form: https://github.com/yhenon/pytorch-retinanet

## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:
	
```
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests

```

## Training

The network can be trained using the `train.py` script. Use the CSV dataloader. For training, use


OMP_NUM_THREADS=1 python train.py --batch_size=4 --dataset csv --csv_train train2017.csv  --csv_classes classes.csv  --csv_val valid2017.csv 



## Pre-trained model

A pre-trained model is available at: 
- https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS (this is a pytorch state dict)

The state dict model can be loaded using:

```
retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
```
