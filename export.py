import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# some wrapper class for export
class Embedding(torch.nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, input_ids):
        return self.embed(input_ids)

class Block(torch.nn.Module):
    def __init__(self, block, block_id, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.final_layernorm = final_layernorm

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        '''
        # chatglm2
        theta = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).transpose(0, 1).contiguous()
        hidden_states, presents = self.block(hidden_states,
                                            attention_mask,
                                            kv_cache=past_kv,
                                            rotary_pos_emb=rotary_pos_emb)
        
        # chatglm
        hidden_states, presents = self.block(hidden_states,
                                             position_ids,
                                             attention_mask,
                                             self.block_id,
                                             past_kv)
        '''
        # qwen
        hidden_states, presents = self.block(hidden_states,
                                             past_kv,
                                             attention_mask,
                                             use_cache=True)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, presents

class Lm(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, hidden_states):
        m_logits = self.lm(hidden_states)
        token = torch.argmax(m_logits)
        print('m_logits.shape = ', m_logits.shape)
        print('token = ', token)
        return token

class LLM(torch.nn.Module):
    def __init__(self, tokenizer, embed, blocks, final_layernorm, lm):
        super().__init__()
        self.tokenizer = tokenizer
        self.embed = Embedding(embed)
        self.lm = Lm(lm)
        self.block_nums = len(blocks)
        self.blocks = [Block(blocks[i], i, final_layernorm if i == len(blocks) - 1 else None) for i in range(len(blocks))]

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        # hidden_states = self.embed(input_ids).view(-1, 1, 4096)
        hidden_states = self.embed(input_ids).view(1, -1, 4096)
        for i in range(self.block_nums):
            hidden_states, kv = self.blocks[i](hidden_states, attention_mask, position_ids, past_key_values[i])
            past_key_values[i] = kv
        hidden_states = hidden_states.view(-1, 4096)[-1].view(1, 1, 4096)
        token_id = self.lm(hidden_states).view(1)
        return token_id, past_key_values

    def build_prompt(self, query):
        if hasattr(self.tokenizer, 'build_prompt'):
            prompt = self.tokenizer.build_prompt(query)
        else:
            prompt = query
        return prompt

    def str_to_ids(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
        return input_ids, attention_mask

    def id_to_str(self, token_id):
        word = self.tokenizer._convert_id_to_token(int(token_id))
        word = self.tokenizer.convert_tokens_to_string([word])
        return word

    def response(self, query):
        prompt = self.build_prompt(query)
        '''
        query =  你好
        raw_text =  <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant

        input_ids =  tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
                151645,    198, 151644,    872,    198, 108386, 151645,    198, 151644,
                77091,    198]])
        '''
        input_ids, attention_mask = self.str_to_ids(prompt)
        #'''
        input_ids =  torch.tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
                151645,    198, 151644,    872,    198, 108386, 151645,    198, 151644,
                77091,    198]])
        #'''
        token_len = len(input_ids)
        attention_mask = ~torch.tril(torch.ones([1, 1, token_len, token_len]).bool())
        attention_mask = -torch.zeros([token_len], dtype=torch.float32)
        # position_ids = torch.arange(token_len, dtype=torch.long)
        position_ids_0 = torch.arange(token_len, dtype=torch.long)
        position_ids_1 = torch.zeros(token_len, dtype=torch.long)
        position_ids_1[-1] = 1
        position_ids = torch.stack([position_ids_0, position_ids_1]).view(1, 2, -1)
        past_key_values = [None for i in range(self.block_nums)]
        token_id, past_key_values = self.forward(input_ids, attention_mask, position_ids, past_key_values)
        print('past_key_values[0][0].shape = ', past_key_values[0][0].shape)
        print(token_id)
        word = self.id_to_str(token_id)
        print(word, end="", flush=True)
        while token_id > 2 and token_len < 64:
            token_len += 1
            # attention_mask = torch.tensor([[[[False]]]])
            position_ids = torch.tensor([token_len - 1])
            token_id, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            word = self.id_to_str(token_id)
            print(word, end="", flush=True)

# model load functions
def load_chatglm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().eval()
    transformer = model.transformer
    lm = model.lm_head
    embed = transformer.word_embeddings
    blocks = transformer.layers
    final_layernorm = transformer.final_layernorm
    return LLM(tokenizer, embed, blocks, final_layernorm, lm)

def load_chatglm2(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().eval()
    transformer = model.transformer
    lm = transformer.output_layer
    embed = transformer.embedding.word_embeddings
    blocks = transformer.encoder.layers
    final_layernorm = transformer.encoder.final_layernorm
    return LLM(tokenizer, embed, blocks, final_layernorm, lm)

def load_Qwen_7B_Chat(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).float().eval()
    transformer = model.transformer
    lm = model.lm_head
    embed = transformer.wte
    blocks = transformer.h
    final_layernorm = transformer.ln_f
    return LLM(tokenizer, embed, blocks, final_layernorm, lm)

load_funcs = {
    'chatglm-6b' : load_chatglm,
    'chatglm2-6b' : load_chatglm2,
    'Qwen-7B-Chat' : load_Qwen_7B_Chat
}

# exporter class
class LLMExporter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.load_func = load_funcs[self.model_name]
    def load(self):
        self.llm = self.load_func(self.model_path)

    def export_embed(self):
        model = self.llm.embed
        input_ids = torch.tensor([0, 1, 2, 3])
        torch.onnx.export(model, (input_ids),
                        f'./onnx/embedding.onnx',
                        verbose=False,
                        input_names=['input_ids'],
                        output_names=['inputs_embeds'],
                        dynamic_axes={"input_ids": {
                            0: "length"
                        }},
                        do_constant_folding=True,
                        opset_version=15)

    def export_block(self, block_id):
        inputs_embeds = torch.randn((3, 1, 4096))
        attention_mask =  torch.tensor([[[[False,  True,  True],
                                        [False, False,  True],
                                        [False, False, False]]]])
        position_ids = torch.tensor([0, 1, 2])
        past_key_values = torch.zeros((2, 0, 1, 2, 128))
        model = self.llm.blocks(block_id)
        torch.onnx.export(
            model, (inputs_embeds, attention_mask, position_ids, past_key_values),
            f'./onnx/glm_block_{block_id}.onnx',
            verbose=False,
            input_names=[
                'inputs_embeds', 'attention_mask', 'position_ids', 'past_key_values'
            ],
            output_names=['hidden_states', 'presents'],
            dynamic_axes={
                "inputs_embeds" : { 0: "seq_len" },
                "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                "position_ids" : { 0: "seq_len" },
                "past_key_values" : { 1: "history_len" }
            },
            do_constant_folding=True,
            opset_version=15)

    def export_lm(self):
        model = self.llm.lm
        input = torch.randn(1, 4096)
        torch.onnx.export(model, (input),
                        f'./onnx/lm_head.onnx',
                        verbose=False,
                        input_names=['hidden_states'],
                        output_names=['token'],
                        do_constant_folding=True,
                        opset_version=15)

    def export_llm(self):
        model = self.llm
        inputs_embeds = torch.randn((3, 1, 4096))
        attention_mask =  torch.tensor([[[[False,  True,  True],
                                        [False, False,  True],
                                        [False, False, False]]]])
        position_ids = torch.tensor([0, 1, 2])
        past_key_values = torch.zeros((2, 0, 1, 2, 128))
        torch.onnx.export(
            model, (inputs_embeds, attention_mask, position_ids, past_key_values),
            f'./onnx/llm.onnx',
            verbose=False,
            input_names=[
                'inputs_embeds', 'attention_mask', 'position_ids', 'past_key_values'
            ],
            output_names=['hidden_states', 'presents'],
            dynamic_axes={
                "inputs_embeds" : { 0: "seq_len" },
                "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                "position_ids" : { 0: "seq_len" },
                "past_key_values" : { 1: "history_len" }
            },
            do_constant_folding=True,
            opset_version=15)

    def test(self, query):
        self.llm.response(query)

if __name__ == '__main__':
    model = '../Qwen-7B-Chat'
    # model = '../chatglm-6b'
    llm_exporter = LLMExporter(model)
    llm_exporter.load()
    llm_exporter.test('你好')
    # llm_exporter.export_block(0)
