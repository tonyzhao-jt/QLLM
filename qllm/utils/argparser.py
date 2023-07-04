# simple argparser for example and test file usage
import os
import argparse
def model_config_argparser():
    root_folder = os.environ.get('ROOT_DIR') if os.environ.get('ROOT_DIR') else '/workspace/qpipe'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='125m')
    parser.add_argument('--model-name', type=str, default='opt')
    args = parser.parse_args()
    return args

def model_sample_gen_argparser():
    root_folder = os.environ.get('ROOT_DIR') if os.environ.get('ROOT_DIR') else '/workspace/qpipe'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='125m')
    parser.add_argument('--model-name', type=str, default='opt')
    # add bitwidth, shard_number
    parser.add_argument('--bitwidth', default=16)
    parser.add_argument('--num-shards', type=int, default=2)
    # num_tokens_to_generate = 10, max_prompt_length = 20
    parser.add_argument('--max-gen-tokens', type=int, default=10)
    parser.add_argument('--max-prompt-length', type=int, default=20)
    args = parser.parse_args()

    bitwidth = args.bitwidth
    if type(bitwidth) is not int and bitwidth.isnumeric():
        args.bitwidth = int(bitwidth)
    return args