# ReFusion
Codes for ***ReFusion: Learning Image Fusion from Reconstruction with Learnable Loss via Meta-Learning (IJCV 2024)***

[Haowen Bai](), [Zixiang Zhao](https://zhaozixiang1228.github.io/), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang), [Yichen Wu](), [Lilun Deng](), [Yukun Cui](), [Baisong Jiang](), [Shuang Xu](https://shuangxu96.github.io/).

-[*[Paper]*](https://link.springer.com/article/10.1007/s11263-024-02256-8)  
-[*[ArXiv]*](https://arxiv.org/abs/2312.07943)  

## Update
- [2025/1] Release inference code.

## Citation

```
@article{bai2024refusion,
  title={ReFusion: Learning Image Fusion from Reconstruction with Learnable Loss Via Meta-Learning},
  author={Bai, Haowen and Zhao, Zixiang and Zhang, Jiangshe and Wu, Yichen and Deng, Lilun and Cui, Yukun and Jiang, Baisong and Xu, Shuang},
  journal={International Journal of Computer Vision},
  pages={1--21},
  year={2024},
  publisher={Springer}
}
```

## Abstract

Image fusion aims to combine information from multiple source images into a single one with more comprehensive informational content. Deep learning-based image fusion algorithms face significant challenges, including the lack of a definitive ground truth and the corresponding distance measurement. Additionally, current manually defined loss functions limit the model's flexibility and generalizability for various fusion tasks. To address these limitations, we propose ReFusion, a unified meta-learning based image fusion framework that dynamically optimizes the fusion loss for various tasks through source image reconstruction. Compared to existing methods, ReFusion employs a parameterized loss function, that allows the training framework to be dynamically adapted according to the specific fusion scenario and task. ReFusion consists of three key components: a fusion module, a source reconstruction module, and a loss proposal module. We employ a meta-learning strategy to train the loss proposal module using the reconstruction loss.  This strategy forces the fused image to be more conducive to reconstruct source images, allowing the loss proposal module to generate a adaptive fusion loss that preserves the optimal information from the source images. The update of the fusion module relies on the learnable fusion loss proposed by the loss proposal module. The three modules update alternately, enhancing each other to optimize the fusion loss for different tasks and consistently achieve satisfactory results. Extensive experiments demonstrate that ReFusion is capable of adapting to various tasks, including infrared-visible, medical, multi-focus, and multi-exposure image fusion.

## 🌐 Usage

### ⚙ Network Architecture

Our ReFusion is implemented in ``ReFusion.py``.

### 🏊 Training
The training code is currently being prepared. Stay tuned!

### 🏄 Testing

**1. Pretrained models**

Pretrained models are available in ``'./models/ReFusion_IVIF.pth'`` , ``'./models/ReFusion_MIF.pth'``, ``'./models/ReFusion_MFIF.pth'`` and ``'./models/ReFusion_MEIF.pth'``, which are responsible for the Infrared-Visible Image Fusion (IVIF), Medical Image Fusion (MIF) , Multi-Focus Image Fusion (MFIF), and Multi-Exposure Image Fusion (MEIF), respectively. 

**2. Test cases**

The 'test_cases' folder contains four tasks, each with two examples. 
Running 
```
python test.py
``` 
will fuse these cases, and the fusion results will be saved in the 'test_results' folder.

**3. Test customization**

Modify the variables in 'test.py' as needed: 'task' (the fusion task), 'path_img1' (the path of the first image), 'path_img2' (the path of the second image), and 'path_result' (the path to save the fusion result). Then run
```
python test.py
``` 
and the fusion results will be saved in the 'path_result'.
