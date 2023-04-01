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
This repo is built to power the project QPipe. Please Cite the paper if you find the repo is useful to you.