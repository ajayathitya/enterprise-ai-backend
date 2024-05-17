import os
import uuid
import logging
from sys import stdout
from .templates import *
from time import time
from datetime import datetime
from pytz import timezone
from dotenv import load_dotenv
from operator import itemgetter

from fastapi import FastAPI
from pymongo import MongoClient
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain_core.pydantic_v1 import parse_obj_as
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from .product_config import product_config
from .utils import calculate_tokens
from .objects import ChatRequest, ChatResponse
from langchain.schema import Document
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.exceptions import HTTPException
from langsmith import traceable
from langchain_core.outputs import LLMResult

load_dotenv()

resources:dict={}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(stream=stdout,level=logging.INFO)
    logging.info('****************SYSTEM STARTUP****************')

    client = MongoClient(os.getenv('COSMOSDB_CONNECTION_STRING'))
    database = client['customerservicegpt']
    resources['chatlog_collection'] = database['chatlog']

    resources['model'] = AzureChatOpenAI(
                                            api_key=os.environ["AZURE_OPENAI_API_KEY_SC"], 
                                            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_SC"], 
                                            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"], 
                                            azure_deployment=os.environ["AZURE_OPENAI_GPT_35_TURBO_CHAT_DEPLOYMENT_NAME"],
                                            temperature=0.001,
                                            model_kwargs={"response_format":{"type":"json_object"}}
                                         )
    resources['embeddings_model'] = AzureOpenAIEmbeddings(
                                            api_key=os.environ["AZURE_OPENAI_API_KEY_E2"], 
                                            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_E2"], 
                                            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"], 
                                            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"]
                                        )
    
    resources['classifcation_chain'] = (query_classification_prompt.with_config({'run_name':'QueryClassificationPrompt'}) | resources['model'] | query_classification_parser.with_config({'run_name':'QueryClassificationParser'})).with_config({'run_name':'QueryClassificationChain'})
    resources['query_rewriter_chain'] = (query_rewriter_prompt.with_config({'run_name':'QueryRewriterPrompt'})  | resources['model'] | query_rewriter_parser.with_config({'run_name':'QueryRewriterParser'})).with_config({'run_name':'QueryRewriterChain'})
    resources['query_response_chain'] = (query_response_prompt.with_config({'run_name':'QueryResponsePrompt'})  | resources['model'] | query_response_parser.with_config({'run_name':'ResponseGenerationParser'})).with_config({'run_name':'ResponseGenerationChain'})
    
    vectorstore = vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
                                            connection_string = os.getenv('COSMOSDB_CONNECTION_STRING'),
                                            namespace="customerservicegpt.knowledge",
                                            embedding=resources['embeddings_model']
                                        )
    resources['retriever'] = vectorstore.as_retriever(search_kwargs={'k': 3})

    resources['response_chain'] = (
                                    {"query":itemgetter("query"),
                                        "session_id":itemgetter("session_id"),       
                                        "chat_format_instructions":itemgetter("chat_format_instructions"),
                                        "memory": itemgetter("session_id") | RunnableLambda(get_memory).with_config({'run_name':'GetMemory'})| RunnableLambda(lambda x: x['memory']).with_config({'run_name':'ModifyMemory'}),
                                    }
                                    | RunnablePassthrough().assign(search_query=resources['query_rewriter_chain'] | RunnableLambda(lambda x: x['search_query']).with_config({'run_name':'AssignRewrittenQuery'})).with_config({'run_name':'RewriteQuery'})
                                    | RunnablePassthrough().assign(retrieved = itemgetter("search_query") | resources['retriever'].with_config({'run_name':'AzureCosmosDBRetriever'}) | RunnableLambda(get_relevant_items)).with_config({'run_name':'GetRelevantItems'})
                                    | RunnablePassthrough().assign(context=RunnableLambda(lambda x:x["retrieved"]['context']).with_config({'run_name':'AssignContext'})).assign(context_id=RunnableLambda(lambda x:x["retrieved"]['context_id']).with_config({'run_name':'AssignContextID'})).with_config({'run_name':'ModifyContext'})
                                    | RunnablePassthrough().assign(final_response=resources['query_response_chain']).with_config({'run_name':'GenerateResponse'})
                                    | RunnablePassthrough().assign(answer=RunnableLambda(lambda x:x["final_response"]["answer"]).with_config({'run_name':'AssignAnswer'})).assign(followup_questions=RunnableLambda(lambda x:x["final_response"]["followup_questions"]).with_config({'run_name':'AssignFollowUpQuestions'}))
                                ).with_config({'run_name':'ResponseSequence'})

    yield

    client.close()
    resources.clear()
    logging.info('****************SYSTEM SHUTDOWN****************')

app = FastAPI(title="FourthSquare API",
              description="FourthSquare API helps you enable RAG based application for your organization quickly",
              version="0.1.0",
              contact={
                  "name": "FourthSquare",
                  "email": "info@fourthsquare.com",
                },
                license_info={
                    "name": "Apache 2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
                },
                lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def get_cost(prompt_tokens:int, completion_tokens:int) -> float:
    if resources['model'].deployment_name.startswith('gpt-4'):
        return round(((prompt_tokens/1000)*0.01)+((completion_tokens/1000)*0.03),5)
    
    if resources['model'].deployment_name.startswith('gpt-3'):
        return round(((prompt_tokens/1000)*0.0005)+((completion_tokens/1000)*0.0015),5)

def get_model_name() -> str:
    return resources['model'].deployment_name.replace('-dev','')

def get_memory(session_id:str='test-session')->dict:
    memory_list = resources['chatlog_collection'].find({"session_id": session_id, "class_label": "query"}, {"query": 1, "answer": 1, "_id": 0}).sort([("timestamp", -1)]).limit(3)
    memory_list: list = reversed(list(memory_list))
    memory: list = []

    for item in memory_list:
        memory.append(('human',item['query']))
        memory.append(('ai',item['answer']))
    return {"memory":memory}

def get_relevant_items(docs:list[Document]) -> dict:
    context: str = f'Source 1: \n\n{product_config.about}{product_config.catalog}\n\n'
    source_num: int = 2
    context_id: list = []

    for i in docs:
        if calculate_tokens(context+i.page_content) < 3500:
            context += '\nSource {}: \n\n{}\n'.format(source_num,i.page_content)
            source_num+=1
            context_id.append(i.metadata['filename'])

    return {
        "context":context,
        "context_id":context_id
    }

@app.get("/",status_code=200)
async def default() -> dict:
    return {"status":"healthy"}
    # return RedirectResponse("/docs")

@app.post("/v1/customerservice/")
async def chat_completion(chat_request: ChatRequest) -> ChatResponse:

    if chat_request.query is None or chat_request.query == '' or chat_request.session_id is None:
        raise HTTPException(status_code=400, detail="query and session_id should not be empty")
    
    conversation_id: str = str(uuid.uuid4())
    config: dict = {"metadata": {"conversation_id": conversation_id}}
    start: time = time()

    with get_openai_callback() as cb:
        try:
            response:dict = resources['classifcation_chain'].invoke({"query":chat_request.query},config=config)

        except Exception as e:
            raise HTTPException(status_code=500, detail="Error during query classification: "+str(e))

        if response['class_label'] == "query" and response['subclass_label'] != "greetings":
            try:
                query_response = resources['response_chain'].invoke({"query":chat_request.query,
                                                                    "session_id":chat_request.session_id,
                                                                    "chat_format_instructions":query_response_parser.get_format_instructions()
                                                                    },config=config)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail="Error during response generation: "+str(e))
            response.update(query_response)

        elif response['subclass_label'] == 'greetings':
            response['answer'] = "Hello! How can I assist you today? Whether you have questions, need assistance, or just want to learn more, I'm here to help!"
        
        elif response['class_label'] == 'tangential':
            if response['subclass_label'] == 'unrelated':
                response['answer'] = ("I'm sorry, It seems that your question may be a bit out of scope. "
                        f"I'm tuned to answer questions related only to {product_config.org_name}. "
                        "Is there anything else I can help you with?")
            else:
                response['answer'] = ("I'm sorry, It seems that your question may be a bit unclear. "
                        "Could you please provide more details or clarify your inquiry? "
                        "This will help me better understand your needs and provide a more accurate and helpful response. ")
        end = time()

    response.update({
                    'total_time':round(end-start,2),
                    'prompt_tokens':cb.prompt_tokens,
                    'completion_tokens':cb.completion_tokens,
                    'total_tokens': cb.total_tokens,
                    'total_cost': get_cost(prompt_tokens=cb.prompt_tokens, completion_tokens=cb.completion_tokens),
                    'model_name':get_model_name(),
                    'timestamp':datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S %p"),
                    'session_id':chat_request.session_id,
                    'conversation_id': conversation_id,
                    'query':chat_request.query,
                    'conversation_id':conversation_id
                    })
    
    response = parse_obj_as(ChatResponse, response)
    
    resources['chatlog_collection'].insert_one(response.__dict__)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)