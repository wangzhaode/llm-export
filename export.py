import datetime
import math
import unittest
import torch
import random
import sys
from transformers import AutoModel, AutoTokenizer
from tokenization_chatglm import ChatGLMTokenizer
import pdb
import numpy as np
import os

CHATGLM2_PATH = "../chatglm2-6b"
folder = "./onnx"

origin_model = AutoModel.from_pretrained(CHATGLM2_PATH,
                                         trust_remote_code=True).float()
origin_model.eval()
transformer = origin_model.transformer
MAX_LEN = transformer.seq_length
for param in origin_model.parameters():
    param.requires_grad = False
num_layers = transformer.encoder.num_layers
layers = transformer.encoder.layers
tokenizer = AutoTokenizer.from_pretrained(CHATGLM2_PATH, trust_remote_code=True)


class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_ids):
        return transformer.embedding.word_embeddings(input_ids)


class GlmBlock(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, past_kv):
        '''
        rotary_pos_emb = transformer.rotary_pos_emb(MAX_LEN)[position_ids]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        '''
        theta = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).transpose(0, 1).contiguous()
        hidden_states, presents = self.layer(hidden_states,
                                            None,
                                            kv_cache=past_kv,
                                            rotary_pos_emb=rotary_pos_emb)
        if self.layer_id == len(layers) - 1:
            hidden_states = transformer.encoder.final_layernorm(hidden_states)

        return hidden_states, presents


class LmHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        m_logits = transformer.output_layer(hidden_states)
        token = torch.argmax(m_logits)
        return token


def export_glm_block(layer_id):
    # input
    inputs_embeds = torch.randn((3, 1, 4096))
    # attention_mask = None
    position_ids = torch.tensor([0, 1, 2])
    past_key_values = torch.zeros((2, 0, 1, 2, 128))
    model = GlmBlock(layer_id)
    torch.onnx.export(
        model, (inputs_embeds, position_ids, past_key_values),
        f'./onnx/glm_block_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'inputs_embeds', 'position_ids', 'past_key_values'
        ],
        output_names=['hidden_states', 'presents'],
        dynamic_axes={
            "inputs_embeds" : { 0: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 1: "history_len" }
        },
        do_constant_folding=True,
        opset_version=14)


def export_embedding():
    model = Embedding()
    torch.onnx.export(model, (torch.tensor([0, 1, 2, 3])),
                      f'./onnx/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['inputs_embeds'],
                      dynamic_axes={"input_ids": {
                          0: "length"
                      }},
                      do_constant_folding=True,
                      opset_version=15)


def export_lm_head():
    model = LmHead()
    input = torch.randn(1, 4096)
    torch.onnx.export(model, (input),
                      f'./onnx/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['token'],
                      do_constant_folding=True,
                      opset_version=15)

def test_net():
    embed = Embedding()
    blocks = [GlmBlock(i) for i in range(num_layers)]
    lm = LmHead()
    # query = '问：你好\n答：'
    query = '你好'
    promt = query
    # promt = tokenizer.build_prompt(query)
    print("prompt = ", promt)
    inputs = tokenizer(promt, return_tensors="pt")
    input_ids = inputs['input_ids']
    token_len = len(input_ids)
    position_ids = inputs['position_ids']

    input_ids = torch.tensor([64790, 64792, 39701])
    position_ids = torch.tensor([0, 1, 2])
    out = embed(input_ids).view(-1, 1, 4096)
    print(out)
    torch.save(out, 'glm_in.pt')
    past_key_values = [None for i in range(num_layers)]
    for i in range(num_layers):
        out, kv_cache = blocks[i](out, position_ids, past_key_values[i])
        print(out)
        past_key_values[i] = kv_cache
    out = out[-1].view(1, 4096)
    print(out)
    # torch.save(out, 'lm_in.pt')
    token = lm(out).view(1)
    print('token = ', token)
    out_ids = [int(token)]
    word = tokenizer._convert_id_to_token(int(token[0]))
    print(word, end="")
    exit(0)
    while token > 2 and token_len < 64:
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, 4096)
        position_ids = torch.tensor([token_len - 1])
        for i in range(num_layers):
            out, kv_cache = blocks[i](out, position_ids, past_key_values[i])
            past_key_values[i] = kv_cache
        token = lm(out).view(1)
        word = tokenizer._convert_id_to_token(int(token[0]))
        print(word, end="")


#test_net_with_mask()

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

#export models
#     # convert_glm_block(i)
# convert_embedding()
# convert_lm_head()

def test_glm_block(index):
    hidden_states = torch.load('glm_in.pt')
    position_ids = torch.tensor([0, 1, 2])
    past_kv = torch.zeros((2, 0, 1, 2, 128))
    layer_id = 0
    model = GlmBlock(layer_id)
    out = model.forward(hidden_states, position_ids, past_kv)
    print(out[0])

def export_embed():
    embed = transformer.embedding.word_embeddings.weight
    print('embed.shape = ', embed.shape)
    import numpy as np
    embed_npy = embed.numpy()
    print('embed_npy = ', embed_npy)
    embed_npy.tofile('slim_embedding.bin')

def export_vocab():
    vocab = tokenizer.get_vocab()
    fp = open('vocab.txt', 'wt')
    for val in vocab.values():
        fp.write(val) 
    fp.close()
    
def export_glms():
    for i in range(num_layers):
        print("convert_block_{}".format(i))
        export_glm_block(i)
    

if __name__ == '__main__':
    # export_glm_block(0)
    export_glms()
    # test_glm_block(0)
    # export_lm_head()
    # export_vocab()
    # test_net()
