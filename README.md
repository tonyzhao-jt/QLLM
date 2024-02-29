# QLLM
QLLM provides
1. Quantized version of open-sourced LLMs
2. Sequential Executed version of open-sourced LLMs

# WHY Quant and Sequential
Due to well-known reasons, the large memory consumption of big models makes it difficult to perform inference on limited GPUs. 
- A simple solution is to temporarily load the memory and execute these layers sequentially.
- And another one is to do compression

# Credit
I refer to many existing LLM implementation provided by ad-hoc researchers, which are
| proj_name | model | repo_link |
|-----------|-------|-----------|
| SmoothQuant | OPT175B | https://github.com/mit-han-lab/smoothquant |
| GPTQ | LLaMa | https://github.com/IST-DASLab/gptq |
| Bloom | Bloom | huggingface-bloom |

# PS
This repo is built to power the project LLM-PQ. Please Cite the paper if you find the repo is useful to you.

our [paper](https://dl.acm.org/doi/10.1145/3627535.3638480):
```bibtex
@inproceedings{10.1145/3627535.3638480,
author = {Zhao, Juntao and Wan, Borui and Wu, Chuan and Peng, Yanghua and Lin, Haibin},
title = {POSTER: LLM-PQ:Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization},
year = {2024},
isbn = {9798400704352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627535.3638480},
doi = {10.1145/3627535.3638480},
pages = {460–462},
keywords = {LM serving, heterogenous cluster, quantization},
series = {PPoPP '24}
}
```