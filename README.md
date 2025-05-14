# YOLO-THERMAL

Tested on Rocky Linux, with Python 3.10, PyTorch 2.1.1, NVIDIA 2080.

### Environment Settings 
* **Install environment using conda**
```
conda create -n yolo_thermal python=3.10.13
conda activate yolo_thermal
```

 * **Install the package**
```
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Download Dataset  
Download thermal dataset for people with mobility restrictions [TD4PWMR] from [Google Drive](https://drive.google.com/file/d/1XxrY23r7UbniAX2mKgi7NvCLt9buQDGd/view?usp=drive_link) or [BaiduYun Drive](https://pan.baidu.com/s/1NcgpMxGXrw6q4PoQCfcoaA?pwd=3650) (Extraction code: 3650) and put it under `./data`.

### Download Trained Weights for Model
Download the trained model weights [yolo-thermal.pt] from [Google Drive](https://drive.google.com/file/d/1tyC8yvdaBvCB7hi_4ul7vG62YtU_Pg-Y/view?usp=drive_link) or [BaiduYun Drive](https://pan.baidu.com/s/1iYnbOF_bZBlDR8k_EVJFNA?pwd=3650) (Extraction code: 3650) and put it under `./weights` for test directly.

### Training and Testing Script 

```
# train
python yolo-thermal_train.py


# evaluate
python yolo-thermal_evaluate.py
```

## Citation
Please kindly cite the paper if the code and dataset are useful and helpful for your research.

    @misc{ni2025thermaldetectionpeoplemobility,
        title={Thermal Detection of People with Mobility Restrictions for Barrier Reduction at Traffic Lights Controlled Intersections}, 
        author={Xiao Ni and Carsten Kuehnel and Xiaoyi Jiang},
        year={2025},
        eprint={2505.08568},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2505.08568}, 
    }