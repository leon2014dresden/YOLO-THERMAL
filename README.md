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
Download tracking datasets
[TD4PWMR](https://drive.google.com/file/d/1XxrY23r7UbniAX2mKgi7NvCLt9buQDGd/view?usp=drive_link)

put it in `./data`.

### Download Trained Weights for Model
Download the trained model weights from [yolo-thermal.pt](https://drive.google.com/file/d/1tyC8yvdaBvCB7hi_4ul7vG62YtU_Pg-Y/view?usp=drive_link) and put it under `./weights` for test directly.

### Training and Testing Script 

```
# train
python yolo-thermal_train.py


# evaluate
python yolo-thermal_evaluate.py
```

## Citation
Please kindly cite the papers if the code and dataset are useful and helpful for your research.

    @misc{Ni2024,
        title={Thermal Detection of People with Mobility Restrictions for Barrier Reduction at Traffic Lights Controlled Intersections},
        author={Xiao Ni and Carsten Kuehnel and Xiaoyi Jiang},
        year={2025},
        eprint={?},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }