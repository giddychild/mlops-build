import streamlit as st
import requests
import os
import json
from utils import generate_html_response
T5_MODEL_URL = os.getenv("T5_MODEL_URL")
# T5_MODEL_URL = "https://qa-t5-bot.kubeflow-playground.gcp.ca-mlops.island.dev.srcd.io/predict"
LLAMA2_MODEL_URL = os.getenv("LLAMA2_MODEL_URL")
class App:
    def __init__(self):
        self.available_collections = ["aws-faqs-bge"]
        self.available_models = ["Flan T5", "LLama2 7B Mini"]
        st.set_page_config(page_title="Cloud Provider RetrievalQA", layout="wide")
        
        if 'collection_selected' not in st.session_state:
            st.session_state['collection_selected'] = None
        if 'model_selected' not in st.session_state:
            st.session_state['model_selected'] = None
        if 'qa' not in st.session_state:
            st.session_state['qa'] = None
        self.sidebar()
    def sidebar(self) -> None:
        with st.sidebar:
            st.markdown(
                "## How to use\n"
                "1. Choose your collection of documents 📄\n"
                "2. Ask a question about your collection\n"
            )
            collection_option = st.selectbox(
                'Choose your collection', self.available_collections, key="collection_option")
            if collection_option:
                if st.session_state.collection_selected != collection_option:
                    st.session_state['collection_selected'] = collection_option
            
            model_option = st.selectbox(
                'Choose your Model', self.available_models, key="model_option")
            if model_option:
                if st.session_state.model_selected != model_option:
                    st.session_state['model_selected'] = model_option
    def run(self):
        def qa_api(self, user_input: str):
            if st.session_state['model_selected'] is "Flan T5":
                url = T5_MODEL_URL
            elif st.session_state['model_selected'] is "Llama2 7B Mini":
                url = LLAMA2_MODEL_URL
                
            post_obj = '{"query": "'+ user_input +'", "collection": "' + st.session_state['collection_selected'] + '"}'
            response = requests.post(url, data = post_obj, headers={"Content-type": "application/json" ,"Accept":"application/json"}, verify=False)
            return(response.text)
        
        collection_selected = st.session_state['collection_selected']
        if not collection_selected:
            st.warning(
                "Choose a collection in the sidebar or crate a new one."
            )
        model_selected = st.session_state['model_selected']
        if not model_selected:
            st.warning(
                "Choose a model in the sidebar or create a new one."
            )
        chain_qa = st.session_state['qa']
        if chain_qa is not 'None':
            user_input = st.text_input(
                "Ask a question about Cloud FQS (AWS/Azure)")
            if user_input:
                with st.spinner("Thinking..."):
                    response = qa_api(self, user_input)
                    # html_response = generate_html_response(user_input, response)
                    response_obj = json.loads(response)
                    with st.chat_message("assistant"):
                        st.write(response_obj['result'])
if __name__ == "__main__":
    app = App()
    app.run()