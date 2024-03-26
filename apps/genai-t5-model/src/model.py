from typing import Dict
from preprocess import QAChain, LSClient, embedding_model, load_model
from kserve import Model, ModelServer
from kserve.errors import InvalidInput


class T5Model(Model):
    def __init__(self):
        name = "flan-t5"
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None
        self.load()
    
    def load(self) -> bool:
        print("Loading model..")
        embedding = embedding_model()
        llm = load_model()
        ls_client = LSClient()
        self.model = QAChain(embedding=embedding, llm=llm, ls=ls_client)
        self.ready = True
    
    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        inputs = payload["instances"]
        query = inputs[0].get("query")
        if query is None:
            raise InvalidInput("The query is required.") 
        
        collection = inputs[0].get("collection")
        if collection is None:
            raise InvalidInput("The collection is required.")
        
        if collection not in self.model.available_collections:
            raise InvalidInput("Invalid collection.")
        
        result = self.model.execute(collection, query)
        return {"predictions": [result]}

if __name__ == "__main__":
    model = T5Model()
    ModelServer().start([model])