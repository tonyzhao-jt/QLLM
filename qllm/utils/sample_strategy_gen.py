from .simple_partition import partition_a_into_b_bins
def create_uniform_sharding_strategies(shards_num, decoder_layer_nums, bitwidth):
    sharding_strategy = {}
    each_layer_shards = partition_a_into_b_bins(decoder_layer_nums, shards_num) # e.g. [8,8,8]
    decoder_layer_range_for_each_shard = []
    for i in range(shards_num):
        decoder_layer_range_for_each_shard.append((sum(each_layer_shards[:i]), sum(each_layer_shards[:i+1])))
    
    for shard in range(shards_num):
        sharding_strategy[shard] = {}
        shard_decoders = decoder_layer_range_for_each_shard[shard]
        for layer in range(shard_decoders[0], shard_decoders[1]):
            sharding_strategy[shard][layer] = {'shard': [0, 1], 'bits': [bitwidth] * 2}
    return sharding_strategy

def create_single_node_sharding_strategies_with_precision_specs(decoder_layer_nums, precision_specs):
    sharding_strategy = {}
    shards_num = 1
    each_layer_shards = partition_a_into_b_bins(decoder_layer_nums, shards_num) # e.g. [8,8,8]
    decoder_layer_range_for_each_shard = []
    for i in range(shards_num):
        decoder_layer_range_for_each_shard.append((sum(each_layer_shards[:i]), sum(each_layer_shards[:i+1])))
    
    for shard in range(shards_num):
        sharding_strategy[shard] = {}
        shard_decoders = decoder_layer_range_for_each_shard[shard]
        for layer in range(shard_decoders[0], shard_decoders[1]):
            sharding_strategy[shard][layer] = {'shard': [0, 1], 'bits': [precision_specs[layer]] * 2}
    return sharding_strategy