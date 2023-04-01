Generation Source code:
- https://github.com/huggingface/transformers/tree/main/src/transformers/generation

Old Generate:
https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py#L101


## use cache
if model has past, then set the past variable to speed up decoding 
https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py#L534
```
if self._use_cache(outputs, use_cache):
    past = outputs[1]
```


## About AT Generate
https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L465
https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L1111