import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

class RecipeGenerator():
    def __init__(self, model_="meta-llama/Llama-2-7b-chat-hf", token=""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_,
                                          use_auth_token=token,)

        self.model = AutoModelForCausalLM.from_pretrained(model_,
                                                    device_map='auto',
                                                    torch_dtype=torch.float16,
                                                    use_auth_token=token
                                                    )

        self.pipe = pipeline("text-generation",
                        model=self.model,
                        tokenizer= self.tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_new_tokens = 512,
                        do_sample=True,
                        top_k=30,
                        num_return_sequences=1,
                        eos_token_id=self.tokenizer.eos_token_id
                        )
        
        self.llm = HuggingFacePipeline(pipeline = self.pipe, model_kwargs = {'temperature':0})
        
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.DEFAULT_SYSTEM_PROMPT = """\
        You must provide the recipe for the asked food, if you don't know the exact recipe, provide recipe of the closest matched food. Include only ingreadient list and recipe steps"""


    def get_prompt(self, instruction, new_system_prompt=""):
        SYSTEM_PROMPT = self.B_SYS + new_system_prompt + self.E_SYS
        prompt_template =  self.B_INST + SYSTEM_PROMPT + instruction + self.E_INST
        return prompt_template

    def generate(self, text="", instruction="Generate the recipe of {text}"):
        system_prompt = self.DEFAULT_SYSTEM_PROMPT
        instruction = instruction
        template = self.get_prompt(instruction, system_prompt)

        prompt = PromptTemplate(template=template, input_variables=["text"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        output = llm_chain.run(text)
        return output

