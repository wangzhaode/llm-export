from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import base64

tokenizer = AutoTokenizer.from_pretrained('../chatglm2-6b', trust_remote_code=True)
vocab = tokenizer.get_vocab()
vocab_list = ['<' + str(i) + '>\n' for i in range(65024)]
for k, v in vocab.items():
    # if len(k) == 0:
    #    k = '<uk>'
    k = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
    vocab_list[v] = k
fp = open('Chatglm2_6b_vocab.txt', 'wt')
for v in vocab_list:
    fp.write(v)
fp.close()