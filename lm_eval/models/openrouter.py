import os
from typing import List, Optional, Union
from openai import OpenAI
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("openrouter")
class OpenRouterLM(LM):
    def __init__(
        self,
        model: str = "meta-llama/llama-4-scout:free",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize OpenAI client for OpenRouter
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        )
        
        # Set up headers
        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self.extra_headers["X-Title"] = site_name
    
    @property
    def eot_token_id(self):
        # This might need adjustment based on the specific model
        return None
    
    @property
    def max_length(self):
        # Return the context length for the model
        # You might need to adjust this based on the specific model
        return 4096
    
    @property
    def max_gen_toks(self):
        return self.max_tokens
    
    @property
    def batch_size(self):
        # OpenRouter API typically handles one request at a time
        return 1
    
    @property
    def device(self):
        return "api"
    
    def tok_encode(self, string: str) -> List[int]:
        # For API models, we typically don't have direct access to tokenization
        # This is a placeholder - you might need to implement this differently
        # or use an approximate tokenizer
        return list(string.encode('utf-8'))
    
    def tok_decode(self, tokens: List[int]) -> str:
        # Corresponding decode function
        return bytes(tokens).decode('utf-8', errors='ignore')
    
    def generate_until(self, requests) -> List[str]:
        """Generate text until stopping criteria are met"""
        if not isinstance(requests, list):
            requests = [requests]
        
        results = []
        for request in requests:
            try:
                # Extract the context and generation arguments
                context = request.args[0] if request.args else ""
                until = request.args[1] if len(request.args) > 1 else []
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": context}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_headers=self.extra_headers,
                    stop=until if until else None,
                )
                
                generated_text = response.choices[0].message.content
                results.append(generated_text)
                
            except Exception as e:
                print(f"Error generating text: {e}")
                results.append("")
        
        return results
    
    def loglikelihood(self, requests) -> List[tuple]:
        """
        Calculate log-likelihood for given requests
        Note: OpenRouter API doesn't typically provide logprobs,
        so this is a simplified implementation
        """
        results = []
        for request in requests:
            # For API models without logprob support, we return dummy values
            # You might want to implement this differently based on your needs
            results.append((0.0, False))  # (logprob, is_greedy)
        
        return results
    
    def loglikelihood_rolling(self, requests) -> List[float]:
        """Rolling window log-likelihood calculation"""
        # Similar limitation as above
        return [0.0] * len(requests)
