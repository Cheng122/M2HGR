# M2HGR: Hand Gesture Recognition for Robotic Teleoperation Using sEMG and Vision Fusion

This is the official implementation of "Hand Gesture Recognition for Robotic Teleoperation Using sEMG and Vision Fusion" on PyTorch platform.


## The released codes include:
    results/:                           the folder for model weights and visualization.
    dataset/:                           the folder for orignal data.
    dataloader/:                        the folder for data loader.
    common/:                            the folder for basic functions.
    models/:                            the folder for networks.
    train_and_test.py:                  the python code for training and testing.

## Environment
Make sure you have the following dependencies installed:
* PyTorch >= 1.3.0
* NumPy
* Matplotlib
* sklearn
* pickle
* tqdm


## Datasets

<p align="center"><img src="RTGesture.jpg" width="100%" alt="" /></p>

- We utilized a proprietary multi-modal dataset (RTGesture) combining visual and sEMG data to train our network. An overview of the RTGesture is illustrated in the figure above.

- The "dataset" directory contains the processed and partitioned multi-modal dataset files: "train.pkl" and "test.pkl".

- We anticipate releasing a comprehensive open-source version of this dataset to the public in the near future. This release will include video data, skeletal information, and sEMG recordings.


## Training and testing
For this stage, please run:
```bash
python train_and_test.py --batch-size 64 --epoch 300
```


## Citation

If you find this repo useful, please consider citing our paper:...

## Acknowledgement
Our code refers to the following repositories. We thank the authors for releasing the codes.

- [DD-Net](https://github.com/BlurryLight/DD-Net-Pytorch) 
- [pyomyo](https://github.com/PerlinWarp/pyomyo)
