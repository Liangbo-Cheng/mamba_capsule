# 💊 Mamba Capsule Routing Towards Part-Whole Relational Camouflaged Object Detection (MCRNet)

![MCRNet](https://github.com/user-attachments/assets/400b892b-633e-4e41-bf39-25686d4b1179)

We introduce the Mamba to generate type-level mamba capsules from the pixel-level capsules for routing, which ensures a lightweight computation, further exploring the part-whole hierarchical relationships in COD.

---

📌 Environmental Setups
---

To set up your environment and install dependencies, run the following commands:

```bash
# Create virtual environment
conda create -n mcr python=3.10
conda activate mcr
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
cd MCRNet
pip install -r requirements.txt
```

📌 Download Datasets
---

Download the camouflaged object detection datasets from [Baidu](https://pan.baidu.com/s/1Fzy4z0gzBMGDBcn2hOSDwA), you can put datasets into the folder 'data'. **PIN:** `ss04`


📌 Checkpoints
---
We offer the training weights of our MCRNet model on [Baidu](https://pan.baidu.com/s/1YLEqlwbjY_Ks6HcMSmq_Cg). **PIN:** `cs28`


📌 Results
---
The prediction maps of our MCRNet can be found on [Baidu](https://pan.baidu.com/s/15wjeefYABaWn5RxiT1QhJg). **PIN:** `l27b`
