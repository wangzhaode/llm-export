import torch
import numpy as np
import onnxruntime as ort
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
        theta = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        # print('position_ids = ', position_ids)
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.cat((idx_theta, idx_theta), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1).unsqueeze(0)
        # rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).transpose(0, 1).contiguous()
        hidden_states, presents = self.block(hidden_states,
                                             past_kv,
                                             attention_mask,
                                             rotary_pos_emb,
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
        #print('m_logits.shape = ', m_logits.shape)
        #print('token = ', token)
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
        presents = []
        for i in range(self.block_nums):
            #torch.save(hidden_states, 'hidden_states.pt')
            #torch.save(attention_mask, 'attention_mask.pt')
            #torch.save(position_ids, 'position_ids.pt')
            #torch.save(past_key_values[i], 'past_key_values.pt')
            hidden_states, kv = self.blocks[i](hidden_states, attention_mask, position_ids, past_key_values[i])
            if past_key_values[i] is not None:
                # print(i, hidden_states)
                # print(i, kv.flatten())
                pass
            # past_key_values[i] = kv
            presents.append(kv)
        hidden_states = hidden_states.view(-1, 4096)[-1].view(1, 1, 4096)
        # print('hidden_states = ', hidden_states)
        token_id = self.lm(hidden_states).view(1)
        presents = torch.stack(presents)
        return token_id, presents

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
        input_ids =  torch.tensor([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
                151645,    198, 151644,    872,    198, 108386, 151645,    198, 151644,
                77091,    198])
        #'''
        token_len = len(input_ids)
        attention_mask = torch.tril(torch.ones([1, 1, token_len, token_len]).bool())
        # position_ids = torch.arange(token_len, dtype=torch.long)
        position_ids = torch.arange(token_len, dtype=torch.long)
        #position_ids_1 = torch.zeros(token_len, dtype=torch.long)
        #position_ids_1[-1] = 1
        #position_ids = torch.stack([position_ids_0, position_ids_1]).view(1, 2, -1)
        past_key_values = [None for i in range(self.block_nums)]
        token_id, past_key_values = self.forward(input_ids, attention_mask, position_ids, past_key_values)
        print('token_id = ', token_id)
        word = self.id_to_str(token_id)
        print(word, end="", flush=True)
        while token_len < 64:
            token_len += 1
            #attention_mask = torch.ones([token_len]).bool()
            attention_mask = torch.ones([1]).bool().reshape([1, 1, 1, 1])
            position_ids = torch.tensor([token_len - 1])
            token_id, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            print('token_id = ', token_id)
            if token_id == self.tokenizer.im_end_id:
                break
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
        self.do_test = True
    def load(self):
        self.llm = self.load_func(self.model_path)
    def export_vocab(self):
        self.llm.tokenizer.save_vocabulary('./')
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
        # test
        original_outs = model(input_ids)
        ort_session = ort.InferenceSession('./onnx/embedding.onnx', providers=['CPUExecutionProvider'])
        inputs = {
            'input_ids' : input_ids.numpy(),
        }
        onnx_outs = ort_session.run(None, inputs)
        self.assert_equal(original_outs, onnx_outs)

    def export_block(self, block_id):
        #'''
        inputs_embeds = torch.randn((1, 3, 4096))
        attention_mask =  torch.tensor([[[[False,  True,  True],
                                        [False, False,  True],
                                        [False, False, False]]]])
        position_ids = torch.tensor([0, 1, 2])
        past_key_values = torch.zeros((2, 1, 0, 32, 128))
        model = self.llm.blocks[block_id]
        torch.onnx.export(
            model, (inputs_embeds, attention_mask, position_ids, past_key_values),
            f'./onnx/glm_block_{block_id}.onnx',
            verbose=True,
            input_names=[
                'inputs_embeds', 'attention_mask', 'position_ids', 'past_key_values'
            ],
            output_names=['hidden_states', 'presents'],
            dynamic_axes={
                "inputs_embeds" : { 1: "seq_len" },
                "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                "position_ids" : { 0: "seq_len" },
                "past_key_values" : { 2: "history_len" }
            },
            do_constant_folding=True,
            opset_version=15)
        if not self.do_test:
            return
        #'''
        def test_block(inputs_embeds, attention_mask, past_key_values):
            block = self.llm.blocks[block_id]
            original_outs = block(inputs_embeds, attention_mask, position_ids, past_key_values)
            ort_session = ort.InferenceSession(f'./onnx/glm_block_{block_id}.onnx', providers=['CPUExecutionProvider'])
            inputs = {
                'inputs_embeds' : inputs_embeds.detach().numpy(),
                'attention_mask' : attention_mask.numpy(),
                'position_ids' : position_ids.numpy(),
                'past_key_values' : past_key_values.numpy()
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)
        test_block(inputs_embeds, attention_mask, past_key_values)
        # after
        inputs_embeds = torch.randn((1, 1, 4096))
        attention_mask =  torch.tensor([[[[True,  True,  True, True]]]])
        position_ids = torch.tensor([3])
        past_key_values = torch.randn((2, 1, 3, 32, 128))
        test_block(inputs_embeds, attention_mask, past_key_values)

        inputs_embeds = torch.randn((1, 20, 4096))
        attention_mask = torch.tril(torch.ones([1, 1, 20, 20]).bool())
        position_ids = torch.arange(20)
        past_key_values = torch.randn((2, 1, 0, 32, 128))
        test_block(inputs_embeds, attention_mask, past_key_values)

    def export_lm(self):
        model = self.llm.lm
        hidden_states = torch.randn(1, 4096)
        torch.onnx.export(model, (hidden_states),
                        f'./onnx/lm_head.onnx',
                        verbose=True,
                        input_names=['hidden_states'],
                        output_names=['token'],
                        do_constant_folding=True,
                        opset_version=15)
        # test lm
        original_outs = model(hidden_states)
        ort_session = ort.InferenceSession('./onnx/lm_head.onnx', providers=['CPUExecutionProvider'])
        inputs = {
            'hidden_states' : hidden_states.numpy(),
        }
        onnx_outs = ort_session.run(None, inputs)
        self.assert_equal(original_outs, onnx_outs)

    def export_llm(self):
        model = self.llm.eval()
        inputs_embeds = torch.tensor([0, 1, 2])
        # attention_mask = torch.tril(torch.ones([1, 1, 3, 3]).bool())
        attention_mask =  torch.tensor([[[[False,  True,  True],
                                        [False, False,  True],
                                        [False, False, False]]]])
        position_ids = torch.tensor([0, 1, 2])
        past_key_values = torch.zeros((32, 2, 1, 0, 32, 128))
        # y = model(inputs_embeds, attention_mask, position_ids, past_key_values)
        torch.onnx.export(
            model, (inputs_embeds, attention_mask, position_ids, past_key_values),
            f'./onnx/llm.onnx',
            verbose=False,
            input_names=[
                'inputs_embeds', 'attention_mask', 'position_ids', 'past_key_values'
            ],
            output_names=['hidden_states', 'presents'],
            dynamic_axes={
                "inputs_embeds" : { 1: "seq_len" },
                "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                "position_ids" : { 0: "seq_len" },
                "past_key_values" : { 2: "history_len" }
            },
            do_constant_folding=True,
            opset_version=15)


    def assert_equal(self, original_outs, onnx_outs):
        if type(original_outs) not in (list, tuple):
            original_outs = (original_outs, )
            onnx_outs = (onnx_outs, )
        same = True
        for orig, onnx in zip(original_outs, onnx_outs):
            orig = orig.detach().numpy()
            if not np.allclose(orig, onnx, rtol=1e-3, atol=1e-3):
                print('Error: onnx outputs dont match original. [shape = {}] onnx: {}, original: {}'.format(onnx.shape, onnx, orig))
                same = False
                break
        if same:
            print('SUCCESS')

    def test_llm(self, query):
        self.llm.response(query)

    def test_block(self, block_id):
        inputs_embeds = torch.load('hidden_states.pt')
        attention_mask = torch.load('attention_mask.pt')
        position_ids = torch.load('position_ids.pt')
        past_key_values = torch.load('past_key_values.pt')
        block = self.llm.blocks[block_id]
        original_outs = block(inputs_embeds, attention_mask, position_ids, past_key_values)
        ort_session = ort.InferenceSession(f'./onnx/glm_block_{block_id}.onnx', providers=['CPUExecutionProvider'])
        inputs = {
            'inputs_embeds' : inputs_embeds.detach().numpy(),
            'attention_mask' : attention_mask.detach().numpy(),
            'position_ids' : position_ids.detach().numpy(),
            'past_key_values' : past_key_values.detach().numpy()
        }
        onnx_outs = ort_session.run(None, inputs)
        print(original_outs[0], onnx_outs[0])
        self.assert_equal(original_outs, onnx_outs)
if __name__ == '__main__':
    model = '../Qwen-7B-Chat'
    # model = '../chatglm-6b'
    llm_exporter = LLMExporter(model)
    llm_exporter.load()
    llm_exporter.test_llm('你好')
    '''
    for i in range(llm_exporter.llm.block_nums):
        if i < 5: continue
        llm_exporter.export_block(i)
        print('export {} block done.'.format(i))
    '''
    # llm_exporter.test_block(0)
    #llm_exporter.export_block(2)
    # llm_exporter.export_lm()
    # llm_exporter.export_embed()
    # llm_exporter.export_llm()
    # llm_exporter.export_vocab()