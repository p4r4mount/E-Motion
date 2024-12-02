# [NeurIPS 2024] E-Motion: Future Motion Simulation via Event Sequence Diffusion

Arvix: [**E-Motion: Future Motion Simulation via Event Sequence Diffusion**](https://arxiv.org/abs/2410.08649).



## 🛠️ Requirements and Installation

* Python >= 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7
* Install required packages:

```bash
git clone https://github.com/p4r4mount/E-Motion.git
cd E-Motion
conda env create -f environment.yml
```



## 🚀 Inference

```bash
python predict.py --model_path /path/to/model/checkpoint \
                  --data_path /path/to/data/file.npy \
                  --output_path /path/to/output/directory
```

