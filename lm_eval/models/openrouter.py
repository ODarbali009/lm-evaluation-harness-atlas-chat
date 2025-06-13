# lm_eval/models/openrouter.py
import openai
from lm_eval.base import BaseLM
import time

class OpenRouterLM(BaseLM):
    def __init__(
        self,
        model="meta-llama/llama-4-scout:free",
        openai_api_key=None,
        base_url="https://openrouter.ai/api/v1",
        request_timeout=60,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=openai_api_key,
        )
        self.request_timeout = request_timeout

    def loglikelihood(self, requests):
        # loglikelihood not implemented for chat models
        raise NotImplementedError("loglikelihood not supported for OpenRouter chat models")

    def generate_until(self, requests):
        res = []
        for context, stop in requests:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": context}],
                )
                output = completion.choices[0].message.content
                res.append((output, len(output)))
            except Exception as e:
                print(f"Error during generation: {e}")
                res.append(("", 0))
                time.sleep(1)
        return res

    @property
    def max_length(self):
        return 4096

    @property
    def max_gen_toks(self):
        return 512

    @property
    def batch_size(self):
        return 1

    @property
    def eos_token_id(self):
        return None
