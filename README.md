<div align='center'>
<h1>HOH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation</h1>

[![Paper](https://img.shields.io/badge/Paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.04800)
[![Dataset](https://img.shields.io/badge/Dataset-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/russwest404/HoH-QAs)
</div>

> This repository contains the code for our **ACL 2025** paper: "**HOH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation**".

## Overview

While Retrieval-Augmented Generation (RAG) has emerged as an effective approach for addressing the knowledge outdating problem in Large Language Models (LLMs), it faces a critical challenge: the prevalence of outdated information in knowledge bases. To address this challenge, we introduce **HoH** (**H**ow **O**utdated Information **H**arms Retrieval-Augmented Generation), the first large-scale benchmark designed to evaluate RAG's robustness against outdated information.

Our benchmark leverages token-level diff algorithms combined with LLM pipelines to efficiently create a large-scale QA dataset that accurately captures temporal knowledge evolution in real-world facts. Through comprehensive experiments, we reveal that outdated information significantly degrades RAG performance by reducing response accuracy and potentially leading to harmful outputs, highlighting the urgent need for innovative solutions to address temporal challenges in RAG.

## Citation

If you use this benchmark in your research, please cite our paper:

```bibtex
@misc{ouyang2025hohdynamicbenchmarkevaluating,
      title={HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation}, 
      author={Jie Ouyang and Tingyue Pan and Mingyue Cheng and Ruiran Yan and Yucong Luo and Jiaying Lin and Qi Liu},
      year={2025},
      eprint={2503.04800},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.04800}, 
}
```
