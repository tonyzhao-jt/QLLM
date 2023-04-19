from .unit_handler import convert_to_unit
class ModelMemEstimator:
    def __init__(self, h1, h2, b, s, n, vocab_size=None, max_position_embeddings=None, word_embed_proj_dim=None) -> None:
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
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.word_embed_proj_dim = word_embed_proj_dim
    
    def calculate_prepost_mem(self, unit='b', bit=16):
        # contain token embedding and positional embedding. Positiona
        if self.vocab_size is None:
            print("Token embedding dim is not specified")
            return 0
        # calculate each embedding size
        # 32 size
        token_embedding_size = self.vocab_size * self.word_embed_proj_dim * 4
        max_pos_embedding_size = self.max_position_embeddings * self.h1 * 4
        # there exists a project_out / project_in for the max_pos_embedding if work_embed_proj_dim != h1
        if self.word_embed_proj_dim != self.h1:
            max_pos_embedding_size += 2 * self.h1 * self.word_embed_proj_dim * 4
        # there could be project_in and out here.
        # lm_head
        lm_head_weight_size = self.vocab_size * self.word_embed_proj_dim * 4
        mem_b = token_embedding_size + max_pos_embedding_size + lm_head_weight_size
        mem_b = mem_b * bit / 32
        mem_b += self.calculate_single_layer_ln_weight() * bit / 16
        return convert_to_unit(mem_b, unit), f"{convert_to_unit(mem_b, unit)} {unit}"
    
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
        # print(self.b, self.h1, (self.s + self.n))
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
                    kv_size = self.calculate_single_layer_maximum_kv_cache() # calculate in fp16
                    # bits
                    bit = bits[idx]
                    if bit == '8:tc': # only for tensorcore, we store the kv in INT8
                        bit = 8
                        kv_size = kv_size * bit / 16 # the feature supported by pure int8
                    all_size_estimation += kv_size 
        return convert_to_unit(all_size_estimation, unit), f'{convert_to_unit(all_size_estimation, unit)} {unit}'
    
    def calculate_temp_embedding_tensor_size(self, unit='b'):
        all = self.b * self.s * (3 * self.h1 + 2 * self.word_embed_proj_dim)
        return convert_to_unit(all, unit), f'{convert_to_unit(all, unit)} {unit}'
    
    def calculate_temp_tensor_size_prefill(self, unit='b'):
        # a layer norm
        attn_ln_tmp_size = self.b * self.s * self.h1 * 2 # by default to 16
        # 3QKV + 1 proj
        qkv_tmp_size = 4 * self.b * self.s * self.h1 * 2 # by default to 16
        # softmax, 32 bit
        softmax_tmp_size = self.b * self.s * self.h1 * 4 # by default to 16
        # 2 BMM (for qk_bmm, there is a softmax)
        bmm_tmp_size = (self.b * self.s * self.s + self.b * self.s * self.h1) * 2 # by default to 16
        # tmp buffer for kv cache
        kv_cache_tmp = 0
        # ffn
        # a layer norm
        ffn_ln_tmp_size = self.b * self.s * self.h1 * 2 # by default to 16
        # activation
        activation_tmp_size = self.b * self.s * self.h2 * 2 # by default to 16
        # fc1 and fc2
        fc_tmp_size = self.b * self.s * (self.h1 + self.h2) * 2 # by default to 16
        # total
        total_tmp_size = attn_ln_tmp_size + qkv_tmp_size + bmm_tmp_size + kv_cache_tmp + softmax_tmp_size + \
              ffn_ln_tmp_size + activation_tmp_size + fc_tmp_size 
        return convert_to_unit(total_tmp_size, unit), f'{convert_to_unit(total_tmp_size, unit)} {unit}'
    
    def calculate_temp_tensor_size_next_i(self, unit='b'):
        # attn
        # a layer norm
        attn_ln_tmp_size = self.b * self.h1 * 2 # by default to 16
        # 3QKV + 1 proj
        qkv_tmp_size = 4 * self.b * self.h1 * 2 # by default to 16
        # 2 BMM (for qk_bmm, there is a softmax)
        bmm_tmp_size = (self.b * (self.s + self.n) + self.b * self.h1) * 2 # by default to 16
        # 32
        softmax_tmp_size = self.b * (self.s + self.n) * 4
        # tmp buffer for kv cache
        kv_cache_tmp = 2 * (self.b * (self.s + self.n) * self.h1) * 2 # by default to 16
        # ffn
        # a layer norm
        ffn_ln_tmp_size = self.b * self.h1 * 2 # by default to 16
        # activation
        activation_tmp_size = self.b * self.h2 * 2 # by default to 16
        # fc1 and fc2
        fc_tmp_size = self.b * (self.h1 + self.h2) * 2 # by default to 16
        # total
        total_tmp_size = attn_ln_tmp_size + qkv_tmp_size + bmm_tmp_size + kv_cache_tmp + softmax_tmp_size + \
              ffn_ln_tmp_size + activation_tmp_size + fc_tmp_size 
        return convert_to_unit(total_tmp_size, unit), f'{convert_to_unit(total_tmp_size, unit)} {unit}'

    # return in bytes
    def calculate_temp_tensor_size(self, unit='b'):
        max_temp = max(self.calculate_temp_tensor_size_prefill(unit)[0], \
                       self.calculate_temp_tensor_size_next_i(unit)[0], \
                        self.calculate_temp_embedding_tensor_size(unit)[0])
        return max_temp, f'{max_temp} {unit}'

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

                