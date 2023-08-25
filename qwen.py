from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('../Qwen-7B-Chat', trust_remote_code=True)

tokenizer.save_vocabulary('./')
