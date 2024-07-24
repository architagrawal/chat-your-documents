
# !pip install openai cohere tiktoken unstructured unstructured[pdf]
# !pip uninstall python-libmagic
# !pip uninstall python-magic
# !pip install python-magic-bin


# import os
# import sys

# import openai
# from langchain.chains import ConversationalRetrievalChain, RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import DirectoryLoader, TextLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain.llms import OpenAI
# from langchain.vectorstores import Chroma

# # import constants

# os.environ["OPENAI_API_KEY"] = "YOUR_KEY"

# # Enable to save to disk & reuse the model (for repeated queries on the same data)
# PERSIST = False

# query = None
# if len(sys.argv) > 1:
#   query = sys.argv[1]

# if PERSIST and os.path.exists("persist"):
#   print("Reusing index...\n")
#   vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
#   index = VectorStoreIndexWrapper(vectorstore=vectorstore)
# else:
#   #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
#   loader = DirectoryLoader("/content/")
#   if PERSIST:
#     index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
#   else:
#     index = VectorstoreIndexCreator().from_loaders([loader])

# chain = ConversationalRetrievalChain.from_llm(
#   llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="YOUR_KEY"),
#   retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
# )

# chat_history = []
# while True:
#   if not query:
#     query = input("Prompt: ")
#   if query in ['quit', 'q', 'exit']:
#     sys.exit()
#   result = chain({"question": query, "chat_history": chat_history})
#   print(result['answer'])

#   chat_history.append((query, result['answer']))
#   query = None


# with gr.Blocks() as demo:

#     chatbot = gr.Chatbot(label='Chat with your data')
#     msg = gr.Textbox()
#     clear = gr.ClearButton([msg, chatbot])

#     msg.submit(create_conversation, [msg, chatbot], [msg, chatbot])

# demo.launch()



#install required packages
!pip install -q transformers peft  openai accelerate bitsandbytes safetensors sentencepiece streamlit chromadb langchain sentence-transformers gradio pypdf

# import dependencies
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

import os
!pip install gradio --upgrade
!pip install typing-extensions
# # Install package

import gradio as gr
from google.colab import drive

import chromadb
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings

openai_api_key  = "Your_key"

# specify model huggingface mode name

# model_name = "anakin87/zephyr-7b-alpha-sharded"
# model_name = "bn22/Mistral-7B-Instruct-v0.1-sharded"
# model_name = "TinyPixel/Llama-2-7B-bf16-sharded"
# model_name = "guardrail/llama-2-7b-guanaco-instruct-sharded"

# function for loading 4-bit quantized model


def load_quantized_model(model_name: str):
    """
    :param model_name: Name or path of the model to be loaded.
    :return: Loaded quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    return model

# fucntion for initializing tokenizer
def initialize_tokenizer(model_name: str):
    """
    Initialize the tokenizer with the specified model_name.

    :param model_name: Name or path of the model for tokenizer initialization.
    :return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer

# !pip install accelerate

# !pip install -i https://test.pypi.org/simple/ bitsandbytes

# load model
# model = load_quantized_model(model_name)

# initialize tokenizer
# tokenizer = initialize_tokenizer(model_name)

# specify stop token ids
stop_token_ids = [0]

# mount google drive and specify folder path
# drive.mount('/content/drive')
# folder_path = '/content/drive/MyDrive/chatpdf/'

folder_path = '/content/'

# !pip install --upgrade --quiet  "unstructured[all-docs]"
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader(folder_path, mode="elements")
# docs = loader.load()
# docs[:5]

# load pdf files
import openai
# from openai.embeddings_utils import get_embedding, cosine_similarity
loader = PyPDFDirectoryLoader(folder_path)
documents = loader.load()

# split the documents in small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) #Chage the chunk_size and chunk_overlap as needed
all_splits = text_splitter.split_documents(documents)
from openai import OpenAI
# specify embedding model (using huggingface sentence transformer)
# embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {"device": "cuda"}
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)
# embeddings = OpenAIEmbeddings()\
# embeddings = get_embedding(product_description, model='text-embedding-ada-002')
# openai_api_key  = "YOUR_KEY"
# os.environ["OPENAI_API_KEY"] = "Your_key"
# client = OpenAI()
# from chroma_embedding_utils import get_embeddings
# response = openai.ChatCompletion.create(
#     model="gpt-4.0-turbo",
#     prompt=all_identical,
#     max_tokens=50  # Adjust as needed
# )
# embeddings = get_embeddings(response["choices"][0]["text"])

# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input = [text], model=model).data[0].embedding

# # df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

# #embed document chunks
# vectordb = Chroma.from_documents(documents=all_splits, embedding=[embeddings], persist_directory="chroma_db")
os.environ["OPENAI_API_KEY"] = "YOUR_KEY"
# !pip install langchain
# from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
# embedding = OpenAIEmbeddings( model='text-embedding-ada-002')
embedding = OpenAIEmbeddings()
!pip install tiktoken
vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory="chroma_db")
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
# specify the retriever
retriever = vectordb.as_retriever()

d = retriever.get_relevant_documents(
    "What is best way for rubrics", k=2
)
# print(d)


from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
# !pip install langchain_openai
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


from langchain.chains.qa_with_sources import load_qa_with_sources_chain
llm=ChatOpenAI(model="gpt-4-0125-preview", openai_api_key="YOUR_KEY")
# doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

# Build prompt
template = """Use the following pieces of context to answer the question at the end.Write a academic research style answer and generate a 3000+ words output.
 If you don't find the answer in context, point out that its not available in provided documents.
 Then analyse the history, context and question , using it to infer an plausible answer.
 Always after each answer say "thanks for asking!" or "Sorry I was not helpfull" at the end of the answer.
####
 ALLWAYS FOLLOW THE TEMPLATE
####
{context}
####
Examples: ("Hi", "Hello. Welcome to QA Bot. I can help with designing the course or making improvemnets."),
("Who is better: Messi Or Ronaldo", "Sorry, I am not yet able to help with this question. Please contact Instruction Designers.")
####
If No context available, infer the motive of question, try to analyse and then develop a context for the question.
####
refer {chat_history} if available
####
Question: {question}
####
Helpful Answer:
####
Finish the answer with "thanks for asking!" or "Sorry I was not helpfull, please connect with ID's" depending if successfull to answer the question or not"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(
     template=template, validate_template=True
)
# memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
doc_chain = load_qa_chain(llm, chain_type="map_reduce")
question_generator_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
print(question_generator_chain)


# question_generator_chain.run(context = d, chat_history = "", human_input = "breif about rubrics")

# build huggingface pipeline for using zephyr-7b-alpha
# !pip install openai
from langchain.chat_models import ChatOpenAI
hist = []
# pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         use_cache=True,
#         device_map="auto",
#         max_length=2048,
#         do_sample=True,
#         top_k=5,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
# )

# specify the llm
# llm = HuggingFacePipeline(pipeline=pipeline)
# !pip install openai
# llm=ChatOpenAI(model="gpt-4-0613", openai_api_key="YOUR_KEY")

# doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

# build conversational retrieval chain with memory (rag) using langchain
def create_conversation(query: str, chat_history: list) -> tuple:
    try:

        # memory = ConversationBufferMemory(
        #     memory_key='chat_history',
        #     return_messages=True
        # )
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            input_key='question',
            output_key='answer',
            memory_key='chat_history',
            return_messages=True
            )
        qa_chain = ConversationalRetrievalChain.from_llm(
    rephrase_question = True,
            llm=llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
            return_source_documents=True,
            return_generated_question = True,
            combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
        )
        # qa_chain= ConversationalRetrievalChain(
        #     combine_docs_chain=doc_chain,
        #     memory=memory,
        #     retriever=vectordb.as_retriever(),
        #     question_generator=question_generator_chain,
        #     get_chat_history=lambda h: h,
        #     return_source_documents=True,
        #     rephrase_question = True,
        #     return_generated_question = True
        # )
        # qa_chain= ConversationalRetrievalChain(

        #     retriever=vectordb.as_retriever(),
        #     question_generator=question_generator_chain,
        #    get_chat_history=lambda h: h,
        #     return_source_documents=True,
        #     combine_docs_chain=doc_chain,

        # )
        # qa_chain = ConversationalRetrievalChain.from_llm(
        #     llm=llm,
        #     retriever=retriever,
        #     memory=memory,
        #     get_chat_history=lambda h: h,
        #     return_source_documents=True,
        #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        # )
        # query = "Is probability a class topic?"
        result = qa_chain.invoke({'question': query, 'chat_history': chat_history})
        hist.append((query, result['answer'], result["source_documents"][0], result['generated_question'][0]))

        chat_history.append((query, result['answer']))
        return '', chat_history


    except Exception as e:
        chat_history.append((query, e))
        return '', chat_history

ideal_qa = [("Hi", "Hello. Welcome to QA Bot. I can help with designing the course or making improvemnets."),
            ("Who is better: Messi Or Ronaldo", "Sorry, I am not yet able to help with this question. Please contact Instruction Designers.")]

# # #experiment section
# from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
# memory = ConversationSummaryBufferMemory(llm=llm, input_key='question', output_key='answer', memory_key='chat_history',
#             return_messages=True)
# # memory = ConversationBufferMemory(
# #         )
# # memory.save_context()
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# # doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
# load_qa_chain(llm, chain_type="map_reduce")
# chat_history = []
# # print(question_generator_chain)

# # qa_chain= ConversationalRetrievalChain(
# #             combine_docs_chain=doc_chain,
# #             memory=memory,
# #             retriever=vectordb.as_retriever(),
# #             question_generator=question_generator_chain,
# #             get_chat_history=lambda h: h,
# #             return_source_documents=True,
# #             rephrase_question = True,
# #             output_key = 'answer',
# #             return_generated_question = True
# #         )
# qa_chain = ConversationalRetrievalChain.from_llm(
#     rephrase_question = True,
#             llm=llm,
#             retriever=retriever,
#             memory=memory,
#             get_chat_history=lambda h: h,
#             return_source_documents=True,
#             return_generated_question = True,
#             combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
#         )
# # print(qa_chain.combine_docs_chain.llm_chai)
# # qa_chain= ConversationalRetrievalChain(
# #          combine_docs_chain=doc_chain,
# #             memory=memory,
# #             retriever=vectordb.as_retriever(),
# #             question_generator=question_generator_chain,
# #             get_chat_history=lambda h: h,
# #          rephrase_question = True,
# #             return_generated_question = True,
# #          return_source_documents=True,
# #          verbose = True
# #         )

# # query = "Explain the context in 5 points"
# query = "If Today is friday, what is tommorow"
# # print(question_generator_chain)
# a = qa_chain.invoke({'question': query, 'chat_history': chat_history})
# # result = qa_chain({'question': query, 'chat_history': chat_history})
# print(a)
# # print(a["source_documents"][0])
# # chat_history.append((query, result['answer']))


# print(a['answer'], a["source_documents"][0], a["generated_question"], sep = "\n")
# # print(a['answer'], a["source_documents"][0], sep = "\n")

# build gradio ui
with gr.Blocks() as demo:

    chatbot = gr.Chatbot(label='Chat with your data')
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(create_conversation, [msg, chatbot], [msg, chatbot])

demo.launch()

print(hist)


import csv

# Example.csv gets created in the current working directory
with open('history.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = "\n")
    my_writer.writerow(hist)

###Things to check out
##router.multi_prompt.multipromptchain
##promptflow by microsoft
##use gptcache -> issue 481
##response schema
##prompt pipeline
##FIX template issues- > not answering acc to template

# https://c3991101c1a2b7b9e9.gradio.live/

