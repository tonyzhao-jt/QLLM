# Usage
By default provides a option to specify the model name and size
- model-name ['opt', 'bloom']
- model-size
e.g. `python3 ref_gen.py --model-name bloom --model-size 560m`
By default, model-name = opt and model-size = 125m

# Reference Sample
We provided a reference batched generation by huggingface's implementation
```bash
    python3 ref_gen.py
```
Feel free to check it with the result provided by the QLLM.

# QLLM Sample
## Fake Calib
To support smoothQuant, we leaves an calibration implementation, there is a future work to make it optinal,
but for the moment, for any model with any size, please run 
```bash
    python3 fake_calib_gen.py --model-name <> --model-size <>
```
before run the QLLM sample
## Bloom Sample Run
```bash
    python3 fake_calib_gen.py --model-name bloom --model-size 560m
    python3 qllm_gen.py --model-name bloom --model-size 560m
    python3 qllm_gen.py --model-name bloom --model-size 560m --bitwidth '8:tc-li'
    python3 qllm_gen.py --model-name bloom --model-size 560m --bitwidth 4 --num-shards 3
```
## OPT Sample Run
For opt, please first convert weight through `weight_converter.py`
- `python3 weight_convert.py --model-name <> --model-size <>`
```bash 
    python3 fake_calib_gen.py --model-size 125m
    python3 qllm_gen.py --model-size 125m
    python3 fake_calib_gen.py --model-size 1.3b
    python3 qllm_gen.py --model-size 1.3b --bitwidth '8:tc-li'
```
## Available Bits
For the moment, select bitwidth from `[3, 4, 8, '8:tc-li', 16]`

## Tensor Parallel:
We acutally support quantized TP for the models, but not provided samples, refer to the tests file in the root.


# Notice
We use the greedy_search as the generation method for the next token, follows
- Huggingface's GenerationMixin @ [src/transformers/generation/utils.py](https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/generation/utils.py#L1155)
- And [greedy](https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/generation/utils.py#L2188) 

