import os
from typing import Dict
from kserve import Model, ModelServer
from vllm import LLM

class vLLMModel(Model):

    def __init__(self, name: str, model_path: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_path = model_path
        self.model = None
        self.load()
    
    def load(self) -> bool:
        print("Loading model..")
        self.model = LLM(model=self.model_path)
        self.ready = True
        return self.model is not None        

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        prompt = payload["prompt"]
        output = self.model.generate(prompt)
        return {"output": output}

if __name__ == "__main__":
    LOCAL_FOLDER = os.getenv("LOCAL_FOLDER")    
    MODEL_NAME = os.getenv("MODEL_NAME")
    local_folder = f"{LOCAL_FOLDER}/{MODEL_NAME}"
    model = vLLMModel("vllm-model", local_folder)
    ModelServer().start([model])