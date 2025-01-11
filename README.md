## Cross-Domain Semantic Segmentation with Continued Migration to Target
 Domain

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

## Improvement
![attention head improvement](resources/cyg1.png)

## Comparison
![Comparison](resources/cyg2.png)

