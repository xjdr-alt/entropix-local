import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# global inports
from entropix.local.config import EntropixConfig

# framework specific imports
from entropix.local.torch.sampler import sample
from entropix.local.torch.dslider import adaptive_dirichlet_step, initialize_state
from entropix.local.torch.dslider_config import get_default_ds_config


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
if device == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")


class EntropixModel:
    def __init__(self, model, seed: int = 1337, dtype: torch.dtype = torch.bfloat16):
        self.model = model
        self.entropix_config = EntropixConfig()
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.vocab_size = self.model.config.vocab_size
        self.dtype = dtype
        self.stop_tokens = torch.tensor(model.config.eos_token_id).to(device, dtype=torch.long)
        if self.stop_tokens.ndim == 0:
            self.stop_tokens = self.stop_tokens.unsqueeze(0)
        # TODO: self.sample_fn

    def generate(self, model_inputs: dict, max_tokens: int = 4096) -> torch.tensor:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        generated_tokens = []

        with torch.inference_mode():
            tokens = model_inputs["input_ids"]
            bsz, seqlen = tokens.shape
            cfg = get_default_ds_config(self.vocab_size).to(device)

            logits = self.model(**model_inputs).logits
            state = initialize_state(logits=logits, bsz=bsz, config=cfg).to(device)
            # next_token, sampler_state = sample(tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
            state, next_token, *_ = adaptive_dirichlet_step(
                key=self.generator, state=state, logits=logits[:, -1], config=cfg
            )
            next_token = next_token.unsqueeze(0)

            gen_tokens = torch.cat((tokens, next_token), dim=1)
            generated_tokens.append(next_token.item())
            cur_pos = seqlen

            while cur_pos < max_tokens:
                model_inputs["input_ids"] = gen_tokens
                logits = self.model(**model_inputs).logits
                # next_token, sampler_state = sample(gen_tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
                state, next_token, *_ = adaptive_dirichlet_step(
                    key=self.generator, state=state, logits=logits[:, -1], config=cfg
                )
                next_token = next_token.unsqueeze(0)
                generated_tokens.append(next_token.item())
                gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                if torch.isin(next_token, self.stop_tokens).any():
                    break

        return generated_tokens

    def generate_stream(self, model_inputs: dict, max_tokens: int = 4096) -> torch.tensor:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.inference_mode():
            tokens = model_inputs["input_ids"]
            bsz, seqlen = tokens.shape
            cfg = get_default_ds_config(self.vocab_size).to(device)

            logits = self.model(**model_inputs).logits
            state = initialize_state(logits=logits, bsz=bsz, config=cfg).to(device)
            # next_token, sampler_state = sample(tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
            state, next_token, *_ = adaptive_dirichlet_step(
                key=self.generator, state=state, logits=logits[:, -1], config=cfg
            )
            next_token = next_token.unsqueeze(0)
            yield next_token

            gen_tokens = torch.cat((tokens, next_token), dim=1)
            cur_pos = seqlen

            while cur_pos < max_tokens:
                model_inputs["input_ids"] = gen_tokens
                logits = self.model(**model_inputs).logits
                # next_token, sampler_state = sample(gen_tokens, logits, scores, self.sampler_config, self.entropix_config, generator=self.generator)
                state, next_token, *_ = adaptive_dirichlet_step(
                    key=self.generator, state=state, logits=logits[:, -1], config=cfg
                )
                next_token = next_token.unsqueeze(0)
                yield next_token

                gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                if torch.isin(next_token, self.stop_tokens).any():
                    break


if __name__ == "__main__":
    #  python3 -m entropix.local.torch.hf_main
    seed = 1337
    torch.manual_seed(seed=seed)

    # Text LLM
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many Rs are there in Strawberry?"},
    ]

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

    entropix_model = EntropixModel(model, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    )

    out = entropix_model.generate(model_inputs)
    out_str = tokenizer.decode(out)
    print(out_str)
    for token in entropix_model.generate_stream(model_inputs):
        print(tokenizer.decode(token[0]), end="", flush=True)

    # VLM
    # model_id = "Qwen/Qwen2-VL-2B-Instruct"
    # from transformers import Qwen2VLForConditionalGeneration
    # from PIL import Image
    # import requests

    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    # )
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    # url = "https://github.com/vikhyat/moondream/blob/main/assets/demo-1.jpg?raw=true"
    # image = Image.open(requests.get(url, stream=True).raw)

    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #             },
    #             {"type": "text", "text": "Describe this image."},
    #         ],
    #     }
    # ]

    # text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")

    # entropix_model = EntropixModel(model, seed=seed)
    # out = entropix_model.generate(inputs)
    # out_str = processor.decode(out)
    # print(out_str)

    # for token in entropix_model.generate_stream(inputs):
    #     print(processor.decode(token[0]), end="", flush=True)
