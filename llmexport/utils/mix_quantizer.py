import gc
import torch
import logging
import inspect
import functools

from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, List, Union, Dict

logging.basicConfig(level=logging.ERROR)

class MixQuantizer:
    def __init__(
        self,
        model,
        modules_to_not_convert=None,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        self.awq_model = model
        self.model = model
        self.tokenizer = model.tokenizer
        self.w_bit = model.args.quant_bit
        self.group_size = model.args.quant_block
        self.zeropoint = not model.args.sym
        self.calib_data = 'ag_news'
        self.split = 'test'
        self.duo_scaling = True
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0
        # zero point quantization
        if self.zeropoint:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            offset = 1 << (self.w_bit - 1)
            clip_max = offset - 1
            clip_min = -offset
            scales = (max_val - min_val) / (clip_max - clip_min)
            zeros =  - torch.round(min_val / scales) + clip_min
            qw = torch.round(w / scales) + zeros
            qw = torch.clamp(qw, clip_min, clip_max)
            w = (qw - zeros) * scales
            zeros = min_val.view(org_w_shape[0], -1)
        else:
            abs_max = w.abs().amax(dim=1, keepdim=True)
            offset = 1 << (self.w_bit - 1)
            clip_max = offset - 1
            clip_min = -clip_max
            scales = abs_max / clip_max
            w = torch.clamp(torch.round(w / scales), clip_min, clip_max)  * scales
            zeros = None

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

    def quantize(self):
        self.init_data()
        self.perplexity()

    def quantize_(self):
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                best_device = MixQuantizer.get_best_device()

                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)
            # print(f'# {i} inps shape: {self.inps.shape}, inps.max: {self.inps.max()}')

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = MixQuantizer.get_named_linears(self.modules[i])
            # Filter out the linear layers we don't want to exclude
            named_linears = MixQuantizer.exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            io_feat = self._get_input_feat(self.modules[i], named_linears)

            MixQuantizer.clear_memory()
            device = next(self.modules[i].parameters()).device
            for name, linear in named_linears.items():
                inp, fp_out = io_feat[name]
                inp = inp.to(device)
                # int8
                self.w_bit = 8
                self.zeropoint = True
                self.group_size = 0
                linear.weight.data = self.pseudo_quantize_tensor(linear.weight.data)[0]
                int8_w_output = self._module_forward(inp, linear, {})
                # int4
                self.w_bit = 4
                self.zeropoint = True
                self.group_size = 0
                linear.weight.data = self.pseudo_quantize_tensor(linear.weight.data)[0]
                int4_w_output = self._module_forward(inp, linear, {})
                int8_loss = self._compute_loss(fp_out, int8_w_output, device)
                int4_loss = self._compute_loss(fp_out, int4_w_output, device)
                print(f'layer_{i}.{name} quant loss: {int8_loss}, {int4_loss}')

            MixQuantizer.clear_memory()

    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            # print(module, x, module_kwargs); exit(0)
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss


    @staticmethod
    def exclude_layers_to_not_quantize(linear_layers, modules_to_not_convert):
        if modules_to_not_convert is None:
            return linear_layers

        filtered_layers = {}
        for name, linear_layer in linear_layers.items():
            if not any(key in name for key in modules_to_not_convert):
                filtered_layers[name] = linear_layer
        return filtered_layers

    @staticmethod
    def get_named_linears(module):
        return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}

    @staticmethod
    def get_op_by_name(module, op_name):
        # get the op by its name relative to the module
        for name, m in module.named_modules():
            if name == op_name:
                return m
        raise ValueError(f"Cannot find op {op_name} in module {module}")

    @staticmethod
    def get_calib_dataset(
        data: Union[str, List[str], List[List[int]]] = "pileval",
        tokenizer=None,
        n_samples=128,
        max_seq_len=512,
        split="train",
        text_column="text",
    ):
        if isinstance(data, str):
            from datasets import load_dataset
            if data == "pileval":
                dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            else:
                dataset = load_dataset(data, split=split)
            # dataset = dataset.shuffle(seed=42)
        elif isinstance(data, list):
            if isinstance(data[0], str):
                dataset = [{text_column: text} for text in data]
            elif isinstance(data[0][0], int):
                dataset = data
            else:
                raise NotImplementedError(
                    "Either pass a string to a huggingface dataset or a list"
                    "that is preprocessed with one sample of text per element"
                    " or a list of list of int for tokenized words."
                )
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )

        samples = []
        n_run = 0
        for data in dataset:
            if isinstance(data, list):
                line_encoded = data
            else:
                line = data[text_column]
                line = line.strip()
                line_encoded = tokenizer.encode(line)
            if len(line_encoded) > max_seq_len:
                continue
            sample = torch.tensor([line_encoded])
            if sample.numel() == 0:
                continue
            samples.append(sample)
            n_run += 1
            if n_run == n_samples:
                break
        # now concatenate all samples and split according to max sequence length
        cat_samples = torch.cat(samples, dim=1)
        n_split = cat_samples.shape[1] // max_seq_len
        logging.debug(f" * Split into {n_split} blocks")
        return [
            cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
        ]

    @staticmethod
    def get_best_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"

    @staticmethod
    def clear_memory(weight=None):
        if weight is not None:
            del weight
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def get_op_name(module, op):
        # get the name of the op relative to the module
        for name, m in module.named_modules():
            if m is op:
                return name
        raise ValueError(f"Cannot find op {op} in module {module}")

    @staticmethod
    def append_str_prefix(x, prefix):
        if isinstance(x, str):
            return prefix + x
        elif isinstance(x, tuple):
            return tuple([MixQuantizer.append_str_prefix(y, prefix) for y in x])
        elif isinstance(x, list):
            return [MixQuantizer.append_str_prefix(y, prefix) for y in x]
        else:
            return x

    def init_data(self):
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="test")
        self.ppl_ids = self.tokenizer.encode("\n\n".join(dataset["text"]))

    def perplexity(self):
        stride = 512
        context_length = stride + stride // 2
        seq_len = len(self.ppl_ids)
        criterion = torch.nn.CrossEntropyLoss()
        nlls = []
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + context_length, seq_len)
            chunk_ids = self.ppl_ids[begin_loc:end_loc]
            best_device = MixQuantizer.get_best_device()
            ids = torch.tensor(chunk_ids).to(best_device)
            self.model.model.to(best_device)
            with torch.no_grad():
                outputs = self.model.model(ids)
                print(outputs)
                exit(0)
            # logits = model.forward(chunk_ids)
            # npy_logits = copy.deepcopy(logits.read())
            # logits = torch.from_numpy(npy_logits).squeeze(0)

            target_ids = torch.tensor(chunk_ids)
            trg_len = end_loc - prev_end_loc
            target_ids[:-trg_len] = -100
            neg_log_likelihood = criterion(logits[:-1, :], target_ids[1:])
            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        perplexity = torch.exp(torch.stack(nlls).mean())
        print(f"Perplexity: {perplexity}")

    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.blocks
        samples = MixQuantizer.get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split
        )
        # samples = torch.cat(samples, dim=0)
        samples = torch.cat(samples[:1], dim=0) # just using 1 batch
        inps = []
        layer_kwargs = {}
        # build inps
        self.model.seq_len = samples.numel()
        self.model.context_len = samples.numel() - 2
        self.model.token_len = 0
        best_device = MixQuantizer.get_best_device()
        inps = self.model.embedding(samples).to(best_device)
        position_ids = self.model.get_position_ids()
        rotary_pos_emb = self.model.rotary(position_ids)
        attention_mask = self.model.get_attention_mask()
        layer_kwargs["rotary_pos_emb"] = rotary_pos_emb.to(best_device)
        layer_kwargs["attention_mask"] = attention_mask.to(best_device)
        del samples
        MixQuantizer.clear_memory()
        return modules, layer_kwargs, inps

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            y = y.detach().cpu()
            feat_dict[name] = [x, y]
        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)
        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        # input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs