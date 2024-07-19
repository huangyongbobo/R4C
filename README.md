<div align="center">
<h1>Learning to Rank for Multilabel Aerial Image Classification</h1>

<h5 align="center"><em> Yongbo Huang, Haoran Huang, Ruiru Zhong, Yuanpei Jin, and <a href="http://staff.scnu.edu.cn/yishuliu">Yishu Liu</a>.<sup>*</sup></em></h5>
&emsp;*corresponding author
<h6 </h6>
</div>


<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#usage">Usage</a> |
  <a href="#dataset">Dataset</a> |
  <a href="#visualization">Visualization</a> |
  <a href="#citation">Citation</a>
</p>


## Introduction
This is the official repository of the paper entitled ***"Learning to Rank for Multilabel Aerial Image Classification",***
which provides a PyTorch implementation of R4C. 

## Usage

### Preparation 

```bash
git clone https://github.com/huangyongbobo/R4C.git
cd R4C
pip install -r requirements.txt
```

### Running

```bash
python 
```
Arguments:
- ``--img-path``: 
- ``--pred-only`` 
- ``--grayscale``

For example:
```bash
python 
```

## Dataset

We provide download links for four classification datasets:

| Dataset | Label | Link |
|:-|-:|:-:|
| AID_Multilable | Multi | [Download](https://drive.google.com/drive/folders/1he18p2yNI6IjW_cuT2lRs545pQAG7usZ) |
| UCM_Multilable | Multi | [Download](https://bigearth.eu/datasets) |
| AID | Single | [Download](https://opendatalab.com/OpenDataLab/AID) |
| NWPU-RESISC45 | Single | [Download](https://gcheng-nwpu.github.io/#Datasets) |


## Visualization
### predicted results
Some images and their top 7 predicted probability values. On the left of each image is (are) its positive label(s); besides, false positives and false negatives are highlighted in blue and red, respectively

### Grad-CAM
Grad-CAM visualization of heat maps on multilable dataset. R4C locates the crucial regions more correctly.

## Citation
