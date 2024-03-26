__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import io
import torch
import sys
import chromadb
import subprocess
import gradio as gr
from typing import Optional

from chromadb.config import Settings
from label_studio_sdk import Client

from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM, TextStreamer, pipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
#from huggingface_hub import snapshot_download

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline



LOCAL_FOLDER = os.getenv("LOCAL_FOLDER") #by default, inference services mount on /mnt/pvc
EMBEDDINGS_NAME = os.getenv("EMBEDDINGS_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

class LSClient:
    """
    A client for interacting with Label Studio API.
    Args:
        api_key (str, optional): The API key for Label Studio. Defaults to None.
        url (str, optional): The URL for Label Studio. Defaults to None.
        project_id (int, optional): The ID of the project in Label Studio. Defaults to None.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            url: Optional[str] = None,
            project_id: Optional[int] = None,
    ) -> None:
        if not api_key:
            if os.getenv("LABEL_STUDIO_API_KEY"):
                api_key = str(os.getenv("LABEL_STUDIO_API_KEY"))
            else:
                raise ValueError("Label Studio API key is not provided.")
        if not url:
            if os.getenv("LABEL_STUDIO_URL"):
                url = os.getenv("LABEL_STUDIO_URL")
            else:
                raise ValueError("Label Studio URL is not provided.")
        if not project_id:
            if os.getenv("LABEL_STUDIO_PROJECT_ID"):
                project_id = os.getenv("LABEL_STUDIO_PROJECT_ID")
            else:
                raise ValueError("Label Studio Project ID is not provided.")
        
        self.client = Client(url=url, api_key=api_key)
        self.project = self.client.get_project(project_id)

    def save(self, query, collection, response):
        """
        Saves the given query and response as a task in the project.
        Args:
            query (str): The query to save.
            response (str): The response to save.
        """

        try:
            task = {
                'prompt': query,
                'response': response,
                'collection': collection,
                'model': "llama2-7b",
            }
            self.project.import_tasks(task)
        except Exception as e:
            print("Error saving the task to Label Studio: ", e)

class QAChain:
    """
    A class that represents a QA chain for retrieving answers from a collection of documents.
    Attributes:
        available_collections (list): A list of available collections in the ChromaDB.
        collections_qa (dict): A dictionary that maps collection names to RetrievalQA objects.
        client (chromadb.HttpClient): A ChromaDB HTTP client.
        ls_client (LabelStudioClient): A Label Studio client.
    """

    def __init__(self, embedding, llm, ls=None):
        print("Initialiazing... ")
        self.available_collections = []
        self.collections_qa = {}
        self.client = chromadb.HttpClient(
            host="chromadb.gcp.ca-mlops.island.dev.srcd.io", port=8000,
            settings=Settings(is_persistent=True,
                              persist_directory='/chroma/chroma')
        )

        # Label Studio
        self.ls_client = ls

        for collection in self.client.list_collections():
            self.available_collections.append(collection.name)
            vectordb = Chroma(embedding_function=embedding,
                              client=self.client, collection_name=collection.name)
            
            self.collections_qa[collection.name] = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectordb.as_retriever(),
                chain_type_kwargs={
                    "prompt": self.default_template(),
                },
            )
    
    def default_template(self):
        """
        Returns a default prompt template for the QA chain.
        Returns:
            PromptTemplate: A PromptTemplate object.
        """
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum and keep the answer as concise as possible. 
        Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""

        return PromptTemplate.from_template(template=template)

    def execute(self, collection, query):
        """
        Executes the QA chain for the given collection and query.
        Args:
            collection (str): The name of the collection to search in.
            query (str): The query to search for.
        Returns:
            dict: A dictionary containing the query and the retrieved result.
        """
        chain = self.collections_qa.get(collection)
        result = chain({"query": query})
        if self.ls_client is not None:
            self.ls_client.save(query, collection, result['result'])
        return result


# embedding = None
# tokenizer = None
# model = None
# qa_chain_prompt = None
# available_collections = []

def embedding_model():
    """
    Returns a HuggingFaceBgeEmbeddings object that can be used to encode text into embeddings using the BAAI/bge-base-en model.
    :return: A HuggingFaceBgeEmbeddings object.
    """
    model_name = f"{EMBEDDINGS_NAME}"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    #local_folder = "models"
    #global embedding
    return HuggingFaceBgeEmbeddings(model_name=model_name, cache_folder=LOCAL_FOLDER, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

#def chroma():
    #global client
    #client = chromadb.HttpClient(host="chromadb.gcp.ca-mlops.island.dev.srcd.io", port=8000, settings=Settings(is_persistent=True, persist_directory='/chroma/chroma'))
    #global available_collections
    #for collection in client.list_collections():
     #   available_collections.append(collection.name)


def load_model():
    """
    Loads a pre-trained Hugging Face Llama model and tokenizer from a local folder, and returns a HuggingFacePipeline object
    that can be used to generate text from input prompts.
    Returns:
        HuggingFacePipeline: A pipeline object that can be used to generate text from input prompts.
    """
    #model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"
    #branch = "gptq-4bit-32g-actorder_True"

    #snapshot_download(repo_id=model_name, revision=branch)

    local_folder = f"{LOCAL_FOLDER}/{MODEL_NAME}"
    # model_basename = "gptq_model-4bit-32g"


    use_triton = False

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_folder, use_fast=True)

    global llm
    llm = AutoGPTQForCausalLM.from_quantized(local_folder,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=use_triton,
        inject_fused_attention=False,
        quantize_config=None
    )

    llm_pipeline = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    #global model
    #model = HuggingFacePipeline(pipeline=llm_pipeline, model_kwargs={"temperature": 0})
    return HuggingFacePipeline(pipeline=llm_pipeline, model_kwargs={"temperature": 0})

# def qa_prompt():
#     template = """Use the following pieces of context to answer the question at the end. 
#     If you don't know the answer, just say that you don't know, don't try to make up an answer. 
#     Use three sentences maximum and keep the answer as concise as possible. 
#     Always say "thanks for asking!" at the end of the answer. 
#     {context}
#     Question: {question}
#     Helpful Answer:"""

    # global qa_chain_prompt
    # qa_chain_prompt = PromptTemplate.from_template(template=template)

# def load_qa_chain(collection):
#     vectordb = Chroma(embedding_function=embedding, client=client, collection_name=collection)
#     return RetrievalQA.from_chain_type(
#         llm=model,
#         retriever=vectordb.as_retriever(),
#         chain_type_kwargs={"prompt": qa_chain_prompt}
#     )


#def generate_answer(question):
 #   result = qa_chain({"query": str(question[0])})
  #  return result["result"]

# def preprocessing():
#     embedding_model()
#     load_model()
#     model_pipeline()
#     qa_prompt()
#     chroma()


