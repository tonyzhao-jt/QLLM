
export TRANSFORMERS_CACHE='/data/llms/' # specify transformer cache
# weight conversion
# export LOAD_IN_NP='0' # specify load in checkpoint (CPU mem is abundant)
# python3 weight_convert.py --model-size 125m
# python3 weight_convert.py --model-size 560m --model-name bloom

export LOAD_IN_NP='1' # specify load in numpy
# python3 weight_convert_numpy.py --model-size 125m
# python3 weight_convert_numpy.py --model-size 560m --model-name bloom

# output compare
# python3 output_compare.py --model-size 125m
# python3 output_compare.py --model-size 350m
# python3 output_compare.py --model-size 560m --model-name bloom

# generation test
python3 qllm_gen.py --model-size 125m
# python3 qllm_gen.py --model-size 560m --model-name bloom