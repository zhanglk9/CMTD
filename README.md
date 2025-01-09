## HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation

## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/hrda
source ~/venv/hrda/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Please, download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.
Further, download the checkpoint of [HRDA on GTA→Cityscapes](https://drive.google.com/file/d/1O6n1HearrXHZTHxNRWp8HCMyqbulKcSW/view?usp=sharing) and extract it to the folder `work_dirs/`.

## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia (Optional):** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.


**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## Style transfer
If you want try DGInstyle, please ref [DGInstyle](https://github.com/prs-eth/DGInStyle), and then we filter useful data based on rare classes and lighting conditions, runs:
```shell
python tools/rcs_sample_dg_data.py
python tools/convert_datasets/dg_dataset.py
#!!!!Need to manually modify the file internal path name!!!!
python json_add.py
```
If you want try Basic transfer, please runs:
```shell
python tools/convert_datasets/base_transfer.py
python tools/convert_datasets/gta.py data/basic_style --nproc 8
#!!!!Need to manually modify the file internal path name!!!!
python json_add.py
```


## Training

For convenience, we provide an [annotated config file](configs/hrda/gtaHR2csHR_hrda.py)
of the final HRDA. A training job can be launched using:

```shell
python run_experiments.py --config configs/hrda/gtaHR2csHR_hrda.py
```

The logs and checkpoints are stored in `work_dirs/`.

## Testing & Predictions

The provided HRDA checkpoint trained on GTA→Cityscapes can be tested on the
Cityscapes validation set using:

```shell
sh test.sh work_dirs/gtaHR2csHR_hrda_246ef
```