# MTLink


This repo provides the code for reproducing the experiments in MTLink: Adaptive multi-task learning based pre-trained language model for traceability link recovery between issues and commits. Specially, we propose a novel multi-task learning-based method (i.e., MTLink), which applies the students models to automatically recover the issue-commits links.

The article has been submitted tothe Journal of King Saud University - Computer and Information Sciences.
![2fc07b05-24ad-471e-aad8-f5ea15a3b844](https://github.com/user-attachments/assets/e275dbab-1ff4-4e1a-9745-537521b96318)



## Source Code: 
https://github.com/WTU-intelligent-software-development/MTLink.git

## dataset:

https://drive.google.com/drive/folders/10H0m9WBts52vBoFGc_-spVwP67G2l-s7?usp=drive_link

## Relevant Pretrained-Models

codebert: https://huggingface.co/microsoft/codebert-base

graphcodebert: https://huggingface.co/microsoft/graphcodebert-base

## Run
python==3.9

pip install -r requirements.txt

### Multi-teacher knowledge distillation

run bertdistill_mutiteacher.py

### Adaptive multi-task fine-tuning
run train_ad.py

run test_ad.py


# Reference
If you use this code or MTLink, please consider citing us.
<pre><code>
@article{DENG2024101958,
title = {MTLink: Adaptive multi-task learning based pre-trained language model for traceability link recovery between issues and commits},
journal = {Journal of King Saud University - Computer and Information Sciences},
volume = {36},
number = {2},
pages = {101958},
year = {2024},
issn = {1319-1578},
doi = {https://doi.org/10.1016/j.jksuci.2024.101958},
url = {https://www.sciencedirect.com/science/article/pii/S1319157824000478},
author = {Yang Deng and Bangchao Wang and Qiang Zhu and Junping Liu and Jiewen Kuang and Xingfu Li},
keywords = {Issue-commit link recovery, Multi-teacher knowledge distillation, Adaptive multi-task},
abstract = {Traceability links between issues and commits (issue-commit links recovery (ILR)) play a significant role in software maintenance tasks by enhancing developers’ observability in practice. Recent advancements in large language models, particularly pre-trained models, have improved the effectiveness of automated ILR. However, these models’ large parameter sizes and extended training time pose challenges in large software projects. Besides, existing methods often overlook the association and distinction among artifacts, leading to the generation of erroneous links. To mitigate these problems, this paper proposes a novel link recovery method called MTLink. It utilizes multi-teacher knowledge distillation (MTKD) to compress the model and employs an adaptive multi-task strategy to reduce information loss and improve link accuracy. Experiments are conducted on four open-source projects. The results show that (i) MTLink outperforms state-of-the-art methods; (ii) The multi-teacher knowledge distillation maintains accuracy despite model size reduction; (iii) The adaptive multi-task tracing method effectively handles confusion caused by similar artifacts and balances each task. In conclusion, MTLink offers an efficient solution for ILR in software traceability. The code is available at https://zenodo.org/records/10321150.}
}

  
</code></pre>


