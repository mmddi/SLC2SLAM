# SLC<sup>2</sup>-SLAM: Semantic-guided Loop Closure using Shared Latent Code for NeRF SLAM


### [Paper](https://ieeexplore.ieee.org/document/10935649) | [arXiv](https://arxiv.org/abs/2501.08880)

> SLC<sup>2</sup>-SLAM: Semantic-guided Loop Closure using Shared Latent Code for NeRF SLAM <br />
> [Yuhang Ming](https://yuhangming.github.io/), [Di Ma](https://github.com/mmddi), [Weichen Dai](), [Han Yang](), [Rui Fan](https://www.ruirangerfan.com/), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/), [Wanzeng Kong](https://faculty.hdu.edu.cn/zzb/kwz/main.htm)<br />
> RA-L 2025




<p align="center">
    <img src="https://github.com/mmddi/SLC2SLAM/blob/main/TEASER/0054.png" width="800">
</p>

<p align="center">
    <img src="https://github.com/mmddi/SLC2SLAM/blob/main/TEASER/sys-overview.png" width="800">
</p>


<p>This repository contains the code for the paper SLC<sup>2</sup>-SLAM: Semantic-guided Loop Closure using Shared Latent Code for NeRF SLAM.</p>

## Environment
```bash
# conda environment
conda create -n slcslam python=3.9
conda activate slcslam

# pytorch environment (Please check the cuda version)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 

```

## Run

You can run  SLC<sup>2</sup>-SLAM using the code below:
```bash
python slcslam.py --config './configs/ScanNet/scene0000.yaml'
```
Here, the ScanNet dataset is used as an example; other datasets can be run with similar code.

## Evaluation
Our tracking and reconstruction refer to the code provided by [Co-SLAM](https://github.com/JingwenWang95/neural_slam_eval).

## Acknowledgement
We adapt codes from some awesome repositories, including [Co-SLAM](https://github.com/JingwenWang95/neural_slam_eval),[PyPose](https://github.com/pypose/pypose). Thanks for making the code available

## Citation

If you find our code or paper useful for your research, please consider citing:

```

@ARTICLE{10935649,
  author={Ming, Yuhang and Ma, Di and Dai, Weichen and Yang, Han and Fan, Rui and Zhang, Guofeng and Kong, Wanzeng},
  journal={IEEE Robotics and Automation Letters}, 
  title={SLC$^{2}$-SLAM: Semantic-Guided Loop Closure Using Shared Latent Code for NeRF SLAM}, 
  year={2025},
  pages={1-8},
  keywords={Simultaneous localization and mapping;Neural radiance field;Codes;Tracking loops;Semantics;Feature extraction;Encoding;Optimization;Artificial intelligence;Accuracy;SLAM;Loop Detection;Localization;Semantic Scene Understanding},
  doi={10.1109/LRA.2025.3553352}}
```
