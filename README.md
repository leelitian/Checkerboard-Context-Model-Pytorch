# About
An unofficial pytorch implementation of CVPR2021 paper "Checkerboard Context Model for Efficient Learned Image Compression".

This project is based on CompressAI, **"mbt2018 + checkerboard"** is implemented in `model.py`.

# Usage

## enviroment
python 3.7

compressai 1.2.0

## demo
Due to the limitation of file size in github, you should download checkpoint from Google drive, and then put it into the project fold.

update: Sorry, the checkpoint is lost for some mistakes, please retrain the model using compressai.

```bash
pip install compressai
python demo.py
```

# Reference
https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression

https://github.com/InterDigitalInc/CompressAI

https://github.com/huzi96/Coarse2Fine-PyTorch

Paper: https://arxiv.org/abs/2103.15306

See [my blog](https://blog.csdn.net/leelitian3/article/details/123477382) for more details.
