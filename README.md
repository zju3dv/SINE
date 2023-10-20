# SINE: Semantic-driven Image-based NeRF Editing with Prior-guided Editing Field

### [Project Page](https://zju3dv.github.io/sine/) | [Video](https://www.youtube.com/watch?v=bCovxTtO7vs) | [Paper](https://github.com/chobao/open_access_assets/raw/main/sine/paper.pdf)
<div align=center>
<img src="assets/teaser.gif" width="100%"/>
</div>

> [SINE: Semantic-driven Image-based NeRF Editing with Prior-guided Editing Field](https://github.com/chobao/open_access_assets/raw/main/sine/paper.pdf)  
> 
> [[Chong Bao](https://chobao.github.io/), [Yinda Zhang](https://www.zhangyinda.com/)<sup>Co-Authors</sup>,[Bangbang Yang](https://ybbbbt.com)<sup>Co-Authors</sup>], [Tianxing Fan](https://scholar.google.com/citations?user=siv1RXUAAAAJ&hl=zh-CN), [Zesong Yang](https://github.com/YZsZY), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/),   [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/), [Zhaopeng Cui](https://zhpcui.github.io/). 
> 
> CVPR 2023
> 

⚠️ Note: This is only a preview version of the code. Full code (with training scripts) will be released soon.

## Installation
We have tested the code on Python 3.7.0 and PyTorch 1.10.1, while a newer version of pytorch should also work.
The steps of installation are as follows:
* create virtual environment: `conda create --name sine python=3.7` and activate environment
* install required python packages by `bash install.sh`

## Data
We provide the [poses of each dataset](https://www.dropbox.com/scl/fi/m87bm4a6cgkvymzaecj4w/data.zip?rlkey=13iv03d4blcf2cjj7uz9iasjc&dl=0) for evaluation.

## Evaluation
<!-- We provide [pre-trained models and configs](https://www.dropbox.com/scl/fo/c3pnb6daks5p1872xor8k/h?rlkey=rvifacsdxlp7uziiqhnlzxsg6&dl=0). -->
All the pre-trained models and configs can be found [here](https://www.dropbox.com/scl/fo/c3pnb6daks5p1872xor8k/h?rlkey=rvifacsdxlp7uziiqhnlzxsg6&dl=0).

You can evaluate images with the pre-trained models. 

```python
python eval.py \
    --config configs/texture/vasedeck_snowy.yaml \
    --ckpt_path checkpoints/texture/vasedeck_snowy/latest.ckpt \
    --split test_train 
```


## Citing
```
@inproceedings{bao2023sine,
    title={SINE: Semantic-driven Image-based NeRF Editing with Prior-guided Editing Field},
    author={Bao, Chong and Zhang, Yinda and Yang, Bangbang and Fan, Tianxing and Yang, Zesong and Bao, Hujun and Zhang, Guofeng and Cui, Zhaopeng},
    booktitle={The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
    year={2023}
}
```

## Acknowledgement
In this project we use parts of the implementations of the following works:

* [nerf_pl](https://github.com/kwea123/nerf_pl) by kwea123
* [Splice](https://github.com/omerbt/Splice) by omerbt

We thank the respective authors for open sourcing their methods.