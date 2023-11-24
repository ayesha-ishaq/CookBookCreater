import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import pipeline
from transformers import LlamaTokenizer, AutoModelForCausalLM

class RecipeGenerator():
    def __init__(self, model_ ='lmsys/vicuna-7b-v1.5-16k'):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_)

        self.model = AutoModelForCausalLM.from_pretrained(model_,
                                                    device_map='auto',
                                                    torch_dtype=torch.float16)

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
  

    def generate(self, text="", instruction="Generate the recipe for {text}"):

        prompt = PromptTemplate(template=instruction, input_variables=["text"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        output = llm_chain.run(text)
        return output

