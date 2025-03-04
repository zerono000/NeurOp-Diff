# Continuous Remote Sensing Image Super-Resolution via Neural Operator Diffusion

This repository is the official implementation of the paper "NeurOp-Diff: Continuous Remote Sensing Image Super-Resolution via Neural Operator Diffusion".

# Environment configuration

The codes are based on python 3.11, pytorch 2.4.0 and CUDA version 12.4.

# Data preparation

- [UCMerced](https://drive.google.com/drive/folders/1Mknr0n4VjWIAk3yQwGewplDe4bUa-_D-?usp=drive_link) | [AID_256](https://drive.google.com/drive/folders/1Mknr0n4VjWIAk3yQwGewplDe4bUa-_D-?usp=drive_link) | [RSSCN7_256](https://drive.google.com/drive/folders/1Mknr0n4VjWIAk3yQwGewplDe4bUa-_D-?usp=drive_link)

# checkpoint

The pre-trained weights for continuous SR can be found at this [link](https://drive.google.com/file/d/1A06iFZUyu1-CnYtIceBFmThhdhW65oH8/view?usp=sharing)

# Train

```python main.py --config train.yml --exp ./result --doc pth --timesteps [steps] --ni```

# Test

```python test.py --config ./configs/test.yml --model [checkpoint_path] --timesteps [steps]```

# Demo

```python demo.py --config ./configs/test.yml --model [checkpoint_path] --path [image_path]```

# Acknowledgements

This code is mainly built based on [DDIM](https://github.com/ermongroup/ddim), [SRNO](https://github.com/2y7c3/Super-Resolution-Neural-Operator) and [LIIF](https://github.com/yinboc/liif)