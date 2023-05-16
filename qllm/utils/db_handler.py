def create_ds_indexes(global_bs, micro_bz):
    assert global_bs % micro_bz == 0 # always divisible
    num_micro_bz = global_bs // micro_bz
    batch_indexes = []
    for i in range(num_micro_bz):
        batch_indexes.append([i * micro_bz, (i + 1) * micro_bz])
    return batch_indexes
