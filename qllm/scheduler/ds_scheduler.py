import threading
import torch 

from ..utils import partition_a_with_max_b
class DSScheduler:
    def __init__(self, bz_prefill, decode_bss) -> None:
        self.bz_prefill = bz_prefill
        self.bz_decode = max(decode_bss)
        self.decode_bss = decode_bss
        self.request_table = {}
        self.request_token_generated = {}
        self.num_prefill = 1
        self.done_prefill = []
        self.lock = threading.Lock()
        
        self.request_bs_status = {}
    
    def pass_scheduler(self, request, generated_token):
        self.lock.acquire()
        request_status = self.request_bs_status[request]
        done = request_status['done']
        if done:
            self.lock.release()
            return True, generated_token[:, None]
        else:
            if request not in self.request_token_generated:
                self.request_token_generated[request] = [generated_token[:, None]]
            else:
                self.request_token_generated[request].append(generated_token[:, None])
            process_batch_num = generated_token.shape[0]
            request_status['done_prefill_numbers'] += process_batch_num
            if request_status['done_prefill_numbers'] == request_status['bz']:
                request_status['done'] = True
                self.lock.release()
                return True, torch.cat(self.request_token_generated[request], dim=0)
        self.lock.release()
        return False, False
    
    def split_list_of_prompts(self, list_of_prompts):
        # assert len(list_of_prompts) % self.bz_prefill == 0, "list of prompts must be divisible by split_num"
        assert len(list_of_prompts) == sum(self.decode_bss), "number of prompts is not matched with global batch size"
        for idx, bz in enumerate(self.decode_bss):
            self.request_bs_status[idx] = {
                "bz": bz,
                "done_prefill_numbers": 0,
                "done": False,
                "prefill_batch_size": partition_a_with_max_b(bz, self.bz_prefill)
            }
        current_idx = 0
        request_with_splited_prompts = {}
        for idx, value in self.request_bs_status.items():
            prefill_batch_size = value["prefill_batch_size"]
            splited_prompts = []
            for i in range(len(prefill_batch_size)):
                splited_prompts.append(list_of_prompts[current_idx : current_idx + prefill_batch_size[i]])
                current_idx += prefill_batch_size[i]
            request_with_splited_prompts[idx] = splited_prompts
        return request_with_splited_prompts

    def create_ds_indexes(self):
        assert len(self.request_bs_status) > 0, "must call split_list_of_prompts first"
        request_id_to_batch_indexes = {}
        for idx, value in self.request_bs_status.items():
            prefill_batch_size = value["prefill_batch_size"]
            current_idx = 0
            batch_indexes = []
            for bz in prefill_batch_size:
                batch_indexes.append(torch.tensor([current_idx, current_idx + bz])) 
                current_idx += bz
            request_id_to_batch_indexes[idx] = batch_indexes
        return request_id_to_batch_indexes


    
