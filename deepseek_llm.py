from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pydantic import Field

class DeepSeekR1LLM(LLM):
    model_name: str = "deepseek-ai/DeepSeek-R1"  # default model
    temperature: float = 0.7
    max_new_tokens: int = 200
    device: int = Field(default=-1)  # declare device as a field

    def __init__(self, model_name: str = None, temperature: float = 0.7,
                 max_new_tokens: int = 200, device: int = None, **kwargs):
        super().__init__(**kwargs)  # correctly initialize parent class
        if model_name:
            self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        # Set device using the declared field
        self.__dict__['device'] = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.device != -1:
            self.model.to(f"cuda:{self.device}")
        self.model.eval()

    @property
    def _llm_type(self) -> str:
        return "deepseek-r1"

    def _call(self, prompt: str, stop: list = None) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.device != -1:
            input_ids = input_ids.to(f"cuda:{self.device}")
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if stop:
            for token in stop:
                output = output.split(token)[0]
        return output
