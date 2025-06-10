from abc import ABC, abstractmethod
import os
import json

def get_generator(generator_type, model_name, extra_gen_params={}):
    if generator_type == "openai":
        classname = OpenAI
    elif generator_type == "cohere":
        classname = Cohere
    elif generator_type == "claude":
        classname = Claude
    elif generator_type == "gemini":
        classname = Gemini
    elif generator_type == "hf":
        classname = HF
    else:
         raise ValueError(f"Generator type is not implemented: {generator_type}")
    generator = classname(model_name=model_name,
                          extra_params=eval(extra_gen_params) if extra_gen_params is not None else {},
                         )
    return generator


class Generator(ABC):
    def __init__(self,
                 model_name=None,
                 max_new_tokens=1,
                ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def generate_single(self, input_prompt, system_prompt=""):
        pass

class OpenAI(Generator):
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1, 
                 extra_params={},
                ):
        Generator.__init__(self,
                           model_name=model_name,
                           max_new_tokens=max_new_tokens,
                           )
        import openai
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
        self.extra_params = extra_params

    def create_request(self, prompt, sys_prompt=""):
        request = {
                    "model": self.model_name,
                    "messages":([
                    {
                        'role': 'system',
                        'content': sys_prompt
                    }] if len(sys_prompt) else []) +
                    [{
                        "role": "user",
                        "content": prompt,
                        }
                    ],
                }
        request.update(self.extra_params)
        return request
    
    def generate_single(self, input_prompt, system_prompt=""):
        request = self.create_request(input_prompt, sys_prompt=system_prompt)
        ok = False
        for attempt_idx in range(3):
            chat_completion = self.client.chat.completions.create(**request)
            generated_response = chat_completion.choices[0].message.content
            if generated_response is None or len(generated_response) == 0: continue
            if "json" in str(self.extra_params):
                try: 
                    generated_response = json.loads(generated_response)
                    ok = True
                    break
                except:
                    continue
            else:
                ok = True
                break
        return generated_response, ok

class Cohere(Generator):
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1, 
                 extra_params={},
                ):
        Generator.__init__(self,
                           model_name=model_name,
                           max_new_tokens=max_new_tokens,
                           )
        from cohere import ClientV2
        self.model_name = model_name
        self.client = client = ClientV2(os.environ.get("COHERE_API_KEY"), log_warning_experimental_features=False)
        self.extra_params = extra_params
    
    def generate_single(self, input_prompt, system_prompt=""):
        messages = []
        if system_prompt != "":
            messages.append({
                                "role": "system",
                                "content": system_prompt
                            })
        messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": input_prompt
                            }
                        ]
                    }
        )
        ok = False
        for attempt_idx in range(3):
            response = self.client.chat(
                model = self.model_name,
                messages = messages,
                **self.extra_params
            )
            generated_response = response.message.content[0].text
            if len(generated_response) == 0: continue
            if "json" in str(self.extra_params):
                try: 
                    generated_response = json.loads(generated_response)
                    ok = True
                    break
                except:
                    continue
            else:
                ok = True
                break
        return generated_response, ok

class Claude(Generator):
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1, 
                 extra_params={},
                ):
        Generator.__init__(self,
                           model_name=model_name,
                           max_new_tokens=max_new_tokens,
                           )
        import anthropic
        self.model_name = model_name
        self.client = anthropic.Anthropic()
        if "json" in extra_params:
            self.json = extra_params["json"] # True or False
            extra_params.pop("json")
        else:
            self.json = False
        self.extra_params = extra_params

    def create_request(self, prompt, sys_prompt=""):
        request = {
                    "model": self.model_name,
                    "messages":
                    [{
                        "role": "user",
                        "content": prompt,
                        }
                    ],
                }
        if sys_prompt != "":
            request["system"] = sys_prompt
        request.update(self.extra_params)
        return request
    
    def generate_single(self, input_prompt, system_prompt=""):
        request = self.create_request(input_prompt, sys_prompt=system_prompt)
        ok = False
        for attempt_idx in range(3):
            message = self.client.messages.create(**request)
            generated_response = message.content[0].text
            if generated_response is None or len(generated_response) == 0: continue
            if self.json:
                try: 
                    generated_response = json.loads(generated_response)
                    ok = True
                    break
                except:
                    continue
            else:
                ok = True
                break
        return generated_response, ok

class Gemini(Generator):
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1, 
                 extra_params={},
                ):
        Generator.__init__(self,
                           model_name=model_name,
                           max_new_tokens=max_new_tokens,
                           )
        from google import genai
        self.model_name = model_name
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"),)
        self.extra_params = extra_params

    def create_request(self, prompt, sys_prompt=""):
        request = {
                    "model": self.model_name,
                    "contents": prompt,
                    }
        # todo: system prompt ignored!
        request.update(self.extra_params)
        return request
    
    def generate_single(self, input_prompt, system_prompt=""):
        request = self.create_request(input_prompt, sys_prompt=system_prompt)
        ok = False
        for attempt_idx in range(3):
            response = self.client.models.generate_content(**request)
            generated_response = response.text
            if generated_response is None or len(generated_response) == 0: continue
            if "json" in str(self.extra_params):
                try: 
                    generated_response = json.loads(generated_response)
                    ok = True
                    break
                except:
                    continue
            else:
                ok = True
                break
        return generated_response, ok

class HF(Generator):
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1, 
                 extra_params={},
                ):
        Generator.__init__(self,
                           model_name=model_name,
                           max_new_tokens=max_new_tokens,
                           )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                                                    self.model_name,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map='auto',
                                                )
        self.model.to("cuda")
        self.extra_params = extra_params
    
    def generate_single(self, input_prompt, system_prompt=""):
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt!="" else []) +\
                    [{"role": "user", "content": input_prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = input_ids.to('cuda')
        gen_out = self.model.generate(
            input_ids, 
            **self.extra_params
            )
        
        input_length = input_ids.shape[1]
        response = self.tokenizer.decode(gen_out["sequences"][0, input_length:], skip_special_tokens=True)
        return response, True
