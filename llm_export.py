import os
import base64
import glob
import shutil
import argparse
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# some wrapper class for export
class Embedding(torch.nn.Module):
    def __init__(self, embed, using_bf16: bool = False):
        super().__init__()
        self.bf16 = using_bf16
        if using_bf16:
            # using bf16 embedding weight
            self.embed = embed.bfloat16()
        else:
            self.embed = embed

    def forward(self, input_ids):
        res = self.embed(input_ids)
        if self.bf16:
            res = res.float()
        return res.view(-1, 1, 4096)

class Lm(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, hidden_states):
        m_logits = self.lm(hidden_states)
        token = torch.argmax(m_logits)
        return token

class LLM(torch.nn.Module):
    '''
    Base class for all llm model. Inherits from [`torch.nn.Module`].
    '''

    def __init__(self, args):
        super().__init__()
        self.export_path = args.export_path
        self.export_verbose = args.export_verbose
        self.export_test = args.export_test
        self.embed_bf16 = args.embed_bf16
        self.load_model(args.path)
        self.max_length = 1024

    def load_model(self, model_path: str):
        raise NotImplementedError

    def get_attention_mask(self) -> torch.Tensor:
        raise NotImplementedError

    def get_position_ids(self) -> torch.Tensor:
        raise NotImplementedError

    def export_vocab(self):
        raise NotImplementedError

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        hidden_states = self.embed(input_ids)
        presents = []
        for i in range(self.block_nums):
            hidden_states, kv = self.blocks[i](hidden_states, attention_mask, position_ids, past_key_values[i])
            presents.append(kv)
        token_id = self.lm(hidden_states).view(1)
        presents = torch.stack(presents)
        self.seq_len += 1
        self.token_len += 1
        return token_id, presents

    # some test functions
    def build_prompt(self, query):
        if hasattr(self.tokenizer, 'build_prompt'):
            prompt = self.tokenizer.build_prompt(query)
        else:
            prompt = query
        return prompt

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        return input_ids

    def id_to_str(self, token_id):
        word = self.tokenizer._convert_id_to_token(int(token_id))
        word = self.tokenizer.convert_tokens_to_string([word])
        return word

    def response(self, query):
        prompt = self.build_prompt(query)
        input_ids = self.str_to_ids(prompt)
        self.seq_len = input_ids.numel()
        self.context_len = self.seq_len - 2
        self.token_len = 0
        past_key_values = [None for i in range(self.block_nums)]
        token_id = input_ids
        while self.token_len < self.max_length:
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            token_id, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            if token_id == self.stop_id:
                print("", end='\n')
                break
            word = self.id_to_str(token_id)
            print(word, end="", flush=True)

    # some export functions
    def assert_equal(self, torch_outs, onnx_outs):
        if type(torch_outs) not in (list, tuple):
            torch_outs = (torch_outs, )
            onnx_outs = (onnx_outs, )
        same = True
        for orig, onnx in zip(torch_outs, onnx_outs):
            orig = orig.detach().numpy()
            if not np.allclose(orig, onnx, rtol=1e-3, atol=1e-3):
                print('Error: onnx outputs dont match original. [shape = {}] onnx: {}, original: {}'.format(onnx.shape, onnx, orig))
                same = False
                break
        if same:
            print('onnx test SUCCESS')

    def export_lm(self):
        model = self.lm
        hidden_states = torch.randn(1, 4096)
        onnx_model = f'./{self.export_path}/lm.onnx'
        torch.onnx.export(model, (hidden_states),
                        onnx_model,
                        verbose=self.export_verbose,
                        input_names=['hidden_states'],
                        output_names=['token_id'],
                        do_constant_folding=True,
                        opset_version=15)
        # test lm
        if self.export_test:
            original_outs = model(hidden_states)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'hidden_states' : hidden_states.numpy(),
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)

    def export_embed(self):
        model = self.embed
        input_ids = torch.arange(3, dtype=torch.long)
        onnx_model = f'./{self.export_path}/embedding.onnx'
        torch.onnx.export(model, (input_ids),
                        onnx_model,
                        verbose=self.export_verbose,
                        input_names=['input_ids'],
                        output_names=['inputs_embeds'],
                        dynamic_axes={"input_ids": {
                            0: "length"
                        }},
                        do_constant_folding=True,
                        opset_version=15)
        # test
        if self.export_test:
            original_outs = model(input_ids)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'input_ids' : input_ids.numpy(),
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)

    def export_block(self, block_id: int):
        self.seq_len = 3
        self.token_len = 0
        inputs_embeds = torch.randn((self.seq_len, 1, 4096))
        attention_mask =  self.get_attention_mask()
        position_ids = self.get_position_ids()
        past_key_values = torch.zeros(self.past_kv_shape[1:])
        model = self.blocks[block_id]
        onnx_model = f'./{self.export_path}/block_{block_id}.onnx'
        torch.onnx.export(
            model, (inputs_embeds, attention_mask, position_ids, past_key_values),
            onnx_model,
            verbose=self.export_verbose,
            input_names=[
                'inputs_embeds', 'attention_mask', 'position_ids', 'past_key_values'
            ],
            output_names=['hidden_states', 'presents'],
            dynamic_axes=self.block_dynamic_axes,
            do_constant_folding=True,
            opset_version=15)
        if self.export_test:
            original_outs = model(inputs_embeds, attention_mask, position_ids, past_key_values)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'inputs_embeds' : inputs_embeds.detach().numpy(),
                'attention_mask' : attention_mask.numpy(),
                'position_ids' : position_ids.numpy(),
                'past_key_values' : past_key_values.numpy()
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)

    def export_blocks(self):
        for i in range(self.block_nums):
            self.export_block(i)

    def export(self):
        model = self
        self.seq_len = 3
        self.token_len = 0
        input_ids = torch.arange(3, dtype=torch.long)
        attention_mask =  self.get_attention_mask()
        position_ids = self.get_position_ids()
        past_key_values = torch.zeros(self.past_kv_shape)
        onnx_model = f'./{self.export_path}/llm.onnx'
        torch.onnx.export(
            model, (input_ids, attention_mask, position_ids, past_key_values),
            onnx_model,
            verbose=self.export_verbose,
            input_names=[
                'input_ids', 'attention_mask', 'position_ids', 'past_key_values'
            ],
            output_names=['token_id', 'presents'],
            dynamic_axes=self.model_dynamic_axes,
            do_constant_folding=True,
            opset_version=15)
        if self.export_test:
            # test
            original_outs = model(input_ids, attention_mask, position_ids, past_key_values)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'input_ids' : input_ids.detach().numpy(),
                'attention_mask' : attention_mask.numpy(),
                'position_ids' : position_ids.numpy(),
                'past_key_values' : past_key_values.numpy()
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)

# chatglm
class GLMBlock(torch.nn.Module):
    def __init__(self, block, block_id, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.final_layernorm = final_layernorm

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        hidden_states, presents = self.block(hidden_states,
                                             position_ids,
                                             attention_mask,
                                             self.block_id,
                                             past_kv,
                                             use_cache=True)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, 4096)[-1].view(1, 1, 4096)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        return hidden_states, presents

class Chatglm_6b(LLM):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().eval()
        transformer = model.transformer
        self.lm_ = model.lm_head
        self.embed_ = transformer.word_embeddings
        self.blocks_ = transformer.layers
        self.final_layernorm_ = transformer.final_layernorm
        # some wrapper
        self.stop_id = self.tokenizer._convert_token_to_id(self.tokenizer.eos_token)
        self.block_nums = len(self.blocks_)
        self.lm = Lm(self.lm_)
        # chatglm embedding and lm using same param, copy embedding when using bf16
        if self.embed_bf16:
            import copy
            embed_copy = copy.deepcopy(self.embed_)
            self.embed = Embedding(embed_copy, self.embed_bf16)
        else:
            self.embed = Embedding(self.embed_, self.embed_bf16)
        self.blocks = [GLMBlock(self.blocks_[i], i, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        # some config for export
        self.past_kv_shape = [28, 2, 0, 1, 32, 128]
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 2: "seq_len" },
            "past_key_values" : { 1: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 2: "seq_len" },
            "past_key_values" : { 2: "history_len" }
        }

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1]).bool().reshape([1, 1, 1, 1])
        attention_mask = torch.zeros([self.seq_len, self.seq_len], dtype=torch.bool)
        for i in range(self.seq_len):
            attention_mask[i][-1] = True
        attention_mask = attention_mask.reshape([1, 1, self.seq_len, self.seq_len])
        return attention_mask

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([1, self.seq_len - self.context_len]).reshape([1, 2, 1])
        position_ids_0 = torch.arange(self.seq_len, dtype=torch.long)
        position_ids_1 = torch.zeros(self.seq_len, dtype=torch.long)
        position_ids_1[-1] = 1
        position_ids = torch.stack([position_ids_0, position_ids_1]).view(1, 2, -1)
        return position_ids
    
    def export_vocab(self):
        vocab = self.tokenizer.get_vocab()
        vocab_list = ['' for i in range(len(vocab))]
        for k, v in vocab.items():
            if '<n>' in k: k = '\n'
            if '<|tab|>' in k: k = '\t'
            if '<|blank_' in k: k = ' ' * int(k[8:k.find('|>')])
            if '▁' in k: k = k.replace('▁', ' ')
            k = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
            vocab_list[v] = k
        file_path = os.path.join(self.export_path, "Chatglm_6b_vocab.txt")
        with open(file_path, "w", encoding="utf8") as fp:
            for v in vocab_list:
                fp.write(v)

# chatglm2
class GLM2Block(torch.nn.Module):
    def __init__(self, block, block_id, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.final_layernorm = final_layernorm

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        theta = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).unsqueeze(0).contiguous()
        hidden_states, presents = self.block(hidden_states,
                                            attention_mask,
                                            kv_cache=past_kv,
                                            rotary_pos_emb=rotary_pos_emb)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, 4096)[-1].view(1, 1, 4096)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        return hidden_states, presents

class Chatglm2_6b(LLM):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'Chatglm2_6b'
        if 'codegeex2-6b' in args.path:
            self.model_name = 'Codegeex2_6b'

    def load_model(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().eval()
        transformer = model.transformer
        self.lm_ = transformer.output_layer
        self.embed_ = transformer.embedding.word_embeddings
        self.blocks_ = transformer.encoder.layers
        self.final_layernorm_ = transformer.encoder.final_layernorm
        # some wrapper
        self.stop_id = self.tokenizer.eos_token_id
        if self.stop_id is None:
            # codegeex2-6b
            self.stop_id = self.tokenizer.tokenizer.eos_id
        self.block_nums = len(self.blocks_)
        self.embed = Embedding(self.embed_, self.embed_bf16)
        self.lm = Lm(self.lm_)
        self.blocks = [GLM2Block(self.blocks_[i], i, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        # some config for export
        self.past_kv_shape = [28, 2, 0, 1, 2, 128]
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 1: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 2: "history_len" }
        }

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1, 1, 1, 1]).bool()
        attention_mask = ~torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]).bool())
        return attention_mask

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([self.token_len], dtype=torch.long)
        return torch.arange(self.seq_len, dtype=torch.long)

    def export_vocab(self):
        vocab = self.tokenizer.get_vocab()
        # 65024 is padding size > len(vocab)
        vocab_list = ['<' + str(i) + '>\n' for i in range(65024)]
        for k, v in vocab.items():
            if '▁' in k: k = k.replace('▁', ' ')
            k = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
            vocab_list[v] = k
        file_path = os.path.join(self.export_path, f"{self.model_name}_vocab.txt")
        with open(file_path, "w", encoding="utf8") as fp:
            for v in vocab_list:
                fp.write(v)

# qwen
class QWENBlock(torch.nn.Module):
    def __init__(self, block, block_id, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.final_layernorm = final_layernorm

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        theta = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.cat((idx_theta, idx_theta), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1).unsqueeze(0)
        hidden_states = hidden_states.view(1, -1, 4096)
        hidden_states, presents = self.block(hidden_states,
                                             past_kv,
                                             attention_mask,
                                             rotary_pos_emb,
                                             use_cache=True)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, 4096)[-1].view(1, 1, 4096)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        return hidden_states, presents

class Qwen_7b_Chat(LLM):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).float().eval()
        transformer = model.transformer
        self.lm_ = model.lm_head
        self.embed_ = transformer.wte
        self.blocks_ = transformer.h
        self.final_layernorm_ = transformer.ln_f
        # some wrapper
        self.stop_id = self.tokenizer.im_end_id
        self.block_nums = len(self.blocks_)
        self.embed = Embedding(self.embed_, self.embed_bf16)
        self.lm = Lm(self.lm_)
        self.blocks = [QWENBlock(self.blocks_[i], i, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        # some config for export
        self.past_kv_shape = [32, 2, 1, 0, 32, 128]
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 2: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 3: "history_len" }
        }

    def build_prompt(self, query):
        return f'\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.ones([1, 1, 1, 1]).bool()
        return torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]).bool())

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([self.seq_len - 1], dtype=torch.long)
        return torch.arange(self.seq_len, dtype=torch.long)

    def export_vocab(self):
        file_path = os.path.join(self.export_path, "Qwen_7b_vocab.txt")
        with open(file_path, "w", encoding="utf8") as fp:
            for k, v in self.tokenizer.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + "\n"
                fp.write(line)

# llama2
class LLAMA2Block(torch.nn.Module):
    def __init__(self, block, block_id, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.final_layernorm = final_layernorm

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        hidden_states = hidden_states.view(1, -1, 4096)
        hidden_states, presents = self.block(hidden_states,
                                             attention_mask,
                                             position_ids,
                                             past_kv,
                                             use_cache=True)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, 4096)[-1].view(1, 1, 4096)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        return hidden_states, presents

class Llama2_7b_Chat(LLM):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'Llama2_7b'
        if 'Baichuan2' in args.path:
            self.model_name = 'Baichuan2_7B'

    def load_model(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).float().eval()
        transformer = model.model
        self.lm_ = model.lm_head
        self.embed_ = transformer.embed_tokens
        self.blocks_ = transformer.layers
        self.final_layernorm_ = transformer.norm
        # some wrapper
        self.stop_id = self.tokenizer.eos_token_id
        self.block_nums = len(self.blocks_)
        self.embed = Embedding(self.embed_, self.embed_bf16)
        self.lm = Lm(self.lm_)
        self.blocks = [LLAMA2Block(self.blocks_[i], i, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        # some config for export
        self.past_kv_shape = [32, 2, 1, 32, 0, 128]
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 3: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 4: "history_len" }
        }

    def build_prompt(self, query):
        if 'Baichuan2' in self.model_name:
            return f'<reserved_106>{query}<reserved_107>'
        return f'[INST]{query}[/INST]'


    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1, 1, 1, self.seq_len], dtype=torch.float32)
        return (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([[self.seq_len - 1]], dtype=torch.long)
        return torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)

    def export_vocab(self):
        vocab = self.tokenizer.get_vocab()
        vocab_list = ['<' + str(i) + '>\n' for i in range(len(vocab))]
        for k, v in vocab.items():
            print(k, v)
            if '▁' in k: k = k.replace('▁', ' ')
            k = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
            vocab_list[v] = k
        file_path = os.path.join(self.export_path, f"{self.model_name}_vocab.txt")
        with open(file_path, "w", encoding="utf8") as fp:
            for v in vocab_list:
                fp.write(v)

if __name__ == '__main__':
    llm_models = {
        'chatglm-6b': Chatglm_6b,
        'chatglm2-6b': Chatglm2_6b,
        'codegeex2-6b': Chatglm2_6b,
        'Qwen-7B-Chat': Qwen_7b_Chat,
        'Baichuan2-7B-Chat': Llama2_7b_Chat,
        'Llama-2-7b-chat-ms': Llama2_7b_Chat
    }
    parser = argparse.ArgumentParser(description='LLMExporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path', type=str, default='THUDM/chatglm-6b', required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--type', type=str, choices=llm_models.keys(), default=None,
                        help='type(`str`, *optional*):'
                        '\n\tThe pretrain llm model type.'
                        )
    parser.add_argument('--export_path', type=str, default='./onnx', help='export onnx model path, defaut is `./onnx`.')
    parser.add_argument('--export_verbose', action='store_true', default=False, help='Whether or not to export onnx with verbose.')
    parser.add_argument('--export_test', action='store_true', help='Whether or not to export onnx with test using onnxruntime.')
    parser.add_argument('--test', type=str, help='test model inference with query `TEST`.')
    parser.add_argument('--export', action='store_true', help='export model to an `onnx` model.')
    parser.add_argument('--export_split', action='store_true',
                        help='export model split to some `onnx` models:'
                        '\n\t- embedding model.'
                        '\n\t- block models.'
                        '\n\t- lm_head model.'
                        )
    parser.add_argument('--export_vocab', action='store_true', help='export llm vocab to a txt file.')
    parser.add_argument('--export_embed', action='store_true', help='export llm embedding to an `onnx` model.')
    parser.add_argument('--export_lm', action='store_true', help='export llm lm_head to an `onnx` model.')
    parser.add_argument('--export_block', type=int, help='export llm block [id] to an `onnx` model.')
    parser.add_argument('--export_blocks', action='store_true', help='export llm all blocks to `onnx` models.')
    parser.add_argument('--embed_bf16', action='store_true', help='using `bfloat16` replace `float32` in embedding.')


    args = parser.parse_args()
    model_path = args.path
    model_type = args.type
    # not sepcify model type, using path
    if model_type is None:
        for model in llm_models:
            if model in model_path:
                model_type = model
    if model_type is None:
        raise RuntimeError('Please specify model type.')

    # copy modeling py file to pretrain model for export
    for file in glob.glob(f'./llm_models/{model_type}/*'):
        shutil.copy2(file, model_path)

    llm_exporter = llm_models[model_type](args)

    # some actions
    if args.test is not None:
        llm_exporter.response(args.test)

    if args.export:
        llm_exporter.export()

    if args.export_vocab:
        llm_exporter.export_vocab()

    if args.export_embed or args.export_split:
        llm_exporter.export_embed()

    if args.export_lm or args.export_split:
        llm_exporter.export_lm()

    if args.export_blocks or args.export_split:
        llm_exporter.export_blocks()

    if args.export_block is not None:
        llm_exporter.export_block(args.export_block)
