import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from label_studio_sdk import Client
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline

LOCAL_FOLDER = os.getenv("LOCAL_FOLDER")
EMBEDDINGS_NAME = os.getenv("EMBEDDINGS_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
RESULTS_RETURNED = os.getenv("RESULTS_RETURNED")

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
                'model': "flan-t5",
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
                retriever=vectordb.as_retriever(search_kwargs={"k":int(RESULTS_RETURNED)}),
                return_source_documents=True,
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
        Use five sentences maximum in your answer. 
        {context}
        Question: {question}
        Answer:"""

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


def embedding_model():
    """
    Returns a HuggingFaceBgeEmbeddings object that can be used to encode text into embeddings using the BAAI/bge-base-en model.

    :return: A HuggingFaceBgeEmbeddings object.
    """
    model_name = f"{EMBEDDINGS_NAME}"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceBgeEmbeddings(model_name=model_name, cache_folder=LOCAL_FOLDER, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


def load_model():
    """
    Loads a pre-trained Hugging Face T5 model and tokenizer from a local folder, and returns a HuggingFacePipeline object
    that can be used to generate text from input prompts.

    Returns:
        HuggingFacePipeline: A pipeline object that can be used to generate text from input prompts.
    """
    local_folder = f"{LOCAL_FOLDER}/{MODEL_NAME}"
    tokenizer = AutoTokenizer.from_pretrained(local_folder, use_fast=True)
    llm_pipeline = pipeline(
        "text2text-generation",
        model=local_folder,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    return HuggingFacePipeline(pipeline=llm_pipeline, model_kwargs={"temperature": 0})