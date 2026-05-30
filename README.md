<div align="center">
<h1>A Training-free Co-optimized Conditional Diffusion Model </br> for Multi-source Image Fusion</h1>

[**Ming Lu**](https://luming1314.github.io/), Min Jiang, Shengchen Zhu, Jun Kong <br>

Jiangnan University

<a href='https://doi.org/10.1016/j.optlaseng.2026.109897'><img src='https://img.shields.io/badge/DOI-10.1016%2Fj.optlaseng.2026.109897-blue'></a>

</div>

## ✨ Usage

### Quick start
#### 1. Clone this repo and setting up environment
```sh
git https://github.com/luming1314/cdfm.git
cd cdfm
conda create -n cdfm python=3.8 -y
conda activate cdfm
pip install -r requirements.txt
```

#### 2. Download pre-trained models
* Please download the pre-trained checkpoint `256x256_diffusion_uncond.pt` from [link](https://github.com/openai/guided-diffusion) and place it in `./weight/exp/pre_model`.
* For better reproducibility, we recommend downloading the fixed random noise from [link](https://pan.baidu.com/s/1xkNkXLw5hA8h_gBQEy8H-w?pwd=6vsu) (extraction code: 6vsu) and placing it in `./fixed_seed/`.

#### 3. Run the Demo
Run the demo with the following command:

```shell
sh run_demo.sh
```

#### 4. Testing on All Datasets
Please download the datasets from [this link](https://pan.baidu.com/s/19VaOkEr5enLxnGCBlTjHGg?pwd=fxvq) with the extraction code `fxvq`.

To test all datasets, run the following command:

```shell
sh test_datasets.sh
```

## 👏 Acknowledgment
Our work is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:
* DDFM: https://github.com/Zhaozixiang1228/MMIF-DDFM
* FreeDoM: https://github.com/yujiwen/FreeDoM



## 📬 Contact

For inquiries, please contact **[minglu@stu.jiangnan.edu.cn](mailto:minglu@stu.jiangnan.edu.cn)**.
WeChat: **luming-2077** is also welcomed.

Homepage: https://luming1314.github.io/

## 🎓 Citation

If CDFM is helpful to your work, please cite our paper via:

```bibtex
@article{LU2026109897,
  title = {A training-free co-optimized conditional diffusion model for multi-source image fusion},
  journal = {Optics and Lasers in Engineering},
  volume = {205},
  pages = {109897},
  year = {2026},
  issn = {0143-8166},
  doi = {https://doi.org/10.1016/j.optlaseng.2026.109897},
  url = {https://www.sciencedirect.com/science/article/pii/S0143816626003015},
  author = {Ming Lu and Min Jiang and Shengchen Zhu and Jun Kong},
  keywords = {Infrared-visible, Multi-focus, Multi-exposure, Image fusion, Diffusion model, Deep learning}
}
```
