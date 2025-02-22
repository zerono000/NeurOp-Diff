# Continuous Remote Sensing Image Super-Resolution via Neural Operator Diffusion

This repository is the official implementation of the paper "NeurOp-Diff: Continuous Remote Sensing Image Super-Resolution via Neural Operator Diffusion".

# Environment configuration

The codes are based on python 3.11, pytorch 2.4.0 and CUDA version 12.4.

# Data preparation

- [UCMerced](https://drive.google.com/drive/folders/1Mknr0n4VjWIAk3yQwGewplDe4bUa-_D-?usp=drive_link) | [AID_256](https://drive.google.com/drive/folders/1Mknr0n4VjWIAk3yQwGewplDe4bUa-_D-?usp=drive_link) | [RSSCN7_256](https://drive.google.com/drive/folders/1Mknr0n4VjWIAk3yQwGewplDe4bUa-_D-?usp=drive_link)

# checkpoint

The pre-trained weights for continuous SR can be found at this [link](https://drive.google.com/drive/folders/18bhPVB0V-IzbuUnMToz-IZdObYJpS9fG?usp=drive_link)

# Train

```python main.py --config train.yml --exp ./result --doc pth --timesteps [steps] --ni```

# Test

```python test.py --config ./configs/test.yml --model [checkpoint_path] --timesteps [steps]```

# Demo

```python demo.py --config ./configs/test.yml --model [checkpoint_path] --path [image_path]```