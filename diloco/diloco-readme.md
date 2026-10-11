# DiLoCo on nanoGPT
 
![image](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![image](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![image](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![image](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![image](https://img.shields.io/badge/windows%20terminal-4D4D4D?style=for-the-badge&logo=windows%20terminal&logoColor=white)
 
Author: [Andrew Gyakobo](https://github.com/Gyakobo)
 
A special thanks to [Andrej Karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT) and the [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) series, and to the authors of [DiLoCo](https://arxiv.org/abs/2311.08105) and [OpenDiLoCo](https://arxiv.org/abs/2407.07852), whose work this project reproduces and extends.
 
>[!NOTE]
>This is the research follow-up to my [building-gpt](https://github.com/Gyakobo/building-gpt) project. The code here accompanies a paper being prepared for submission to *Transactions on Machine Learning Research* (TMLR).
 
This project implements **DiLoCo** (*Distributed Low-Communication* training) on top of a small nanoGPT-style language model, and runs a systematic hyperparameter sweep to map how the algorithm's main settings interact at small scale.
 
## Introduction
