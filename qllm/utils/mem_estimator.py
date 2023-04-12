from .unit_handler import convert_to_unit
class ModelMemEstimator:
    def __init__(self, h1, h2, b, s, n) -> None:
        # Refer to the flexGEN
        # h1 hidden dimension
        # h2 hidden dimension of second mlp
        # b batch size
        # s sequence length
        # n generated token numbers
        self.h1 = h1
        self.h2 = h2
        self.b = b
        self.s = s
        self.n = n
        pass

    
    def calculate_single_selfattn_mem(self):
        # QKV storage + OUT projection, 4 linear
        # return bytes
        weight_size = 4 * self.h1 * self.h1 * 4 # 4 means fp32 storage weight has 4 bytes
        return weight_size 

    def calculate_single_FFN_mem(self):
        # 2 linear
        weight_size = 2 * self.h1 * self.h2 * 4
        return weight_size

    def calculate_single_decoder_layer_mem(self):
        # MHA + FFN + 2 linear + 2 layernorm
        # return bytes
        return self.calculate_single_selfattn_mem() + self.calculate_single_FFN_mem() 
    
    def calculate_single_layer_maximum_kv_cache(self):
        size = self.b * self.h1 * (self.s + self.n) * 2 * 2 #(k and v), store in fp16
        return size 
    
    def calculate_single_layer_ln_weight(self):
        size = self.h1 * 2 * 2 # 2 means fp16
        return size
    
    def calculate_multiple_layer_selfattn_mem(self, layer_num):
        return self.calculate_single_selfattn_mem() * layer_num

    def calculate_multiple_layer_FFN_mem(self, layer_num):
        return self.calculate_single_FFN_mem() * layer_num
    
    def calculate_multiple_layer_decoder_layer_mem(self, layer_num):
        return self.calculate_single_decoder_layer_mem() * layer_num
    
    def calculate_multiple_layer_kv_cache(self, layer_num):
        return self.calculate_single_layer_maximum_kv_cache() * layer_num
    
    def calculate_kv_occupation_of_partition(self, partition, unit='b'):
        # partition should be with format
        # {0: {"shard": [0,1], "bits": [8,8]}}
        all_size_estimation = 0
        for layer, config in partition.items():
            shard = config["shard"]
            bits = config["bits"]
            
            if len(shard) != len(bits):
                raise ValueError("shard and bits should have same length")

            for idx, value in enumerate(shard):
                if value == 0:
                    # add the kv size
                    kv_size = self.calculate_single_layer_maximum_kv_cache()
                    # bits
                    bit = bits[idx]
                    if type(bit) != int:
                        bit = 8
                        kv_size = kv_size * bit / 32 # the feature supported by pure int8
                    all_size_estimation += kv_size 
        return convert_to_unit(all_size_estimation, unit), f'{convert_to_unit(all_size_estimation, unit)} {unit}'
    
    def calculate_model_occupation_of_partition(self, partition, unit='b'):
        # partition should be with format
        # {0: {"shard": [0,1], "bits": [8,8]}}
        all_size_estimation = 0
        for layer, config in partition.items():
            shard = config["shard"]
            bits = config["bits"]
            
            if len(shard) != len(bits):
                raise ValueError("shard and bits should have same length")

            for idx, value in enumerate(shard):
                if value == 0:
                    selfattn_mem = self.calculate_single_selfattn_mem()
                    # bits
                    bit = bits[idx]
                    if type(bit) != int:
                        bit = 8
                    selfattn_mem = selfattn_mem * bit / 32 
                    ln_size = self.calculate_single_layer_ln_weight()
                    all_size_estimation += selfattn_mem + ln_size
                elif value == 1:
                    ffn_mem = self.calculate_single_FFN_mem()
                    bit = bits[idx]
                    if type(bit) != int:
                        bit = 8
                    ffn_mem = ffn_mem * bit / 32
                    all_size_estimation += ffn_mem
        return convert_to_unit(all_size_estimation, unit), f'{convert_to_unit(all_size_estimation, unit)} {unit}'

    def calculate_maximum_mem_occupation_of_partition(self, partition, unit='b'):
        # partition should be with format
        # {0: {"shard": [0,1], "bits": [8,8]}}
        all_size_estimation = 0
        kv_mem = self.calculate_kv_occupation_of_partition(partition, unit)[0]
        model_mem = self.calculate_model_occupation_of_partition(partition, unit)[0]
        all_size_estimation = kv_mem + model_mem
        return all_size_estimation, f"{all_size_estimation} {unit}" 
    
    def estimate_hidden_space(self):
        print(self.b, self.s + self.n - 1, self.h1)
        return self.h1 * self.b * (self.s + self.n - 1)

    def estimate_single_layer_kv_cache(self, unit='b'):
        print(self.b, (self.s + self.n - 1), self.h1)
        return self.calculate_single_layer_maximum_kv_cache(), f"{convert_to_unit(self.calculate_single_layer_maximum_kv_cache(), unit)} {unit}"

                