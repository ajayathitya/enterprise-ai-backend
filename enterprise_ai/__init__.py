import os
import uuid
import json
import base64
import random
import logging
import pandas as pd
from sys import stdout
from .templates import *
from time import time
from datetime import datetime
from pytz import timezone
from dotenv import load_dotenv
from operator import itemgetter

from fastapi import FastAPI, UploadFile
from pymongo import MongoClient
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain_core.pydantic_v1 import parse_obj_as
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from .product_config import product_config
from .utils import calculate_tokens
from .objects import ChatRequest, ChatResponse, OcrResponse, AnalyticsResponse
from langchain.schema import Document
from langchain.schema import SystemMessage,HumanMessage
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

load_dotenv()

resources:dict={}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(stream=stdout,level=logging.INFO)
    logging.info('****************SYSTEM STARTUP****************')

    client = MongoClient(os.getenv('COSMOSDB_CONNECTION_STRING'))
    resources['chatlog_collection'] = client['customerservicegpt']['chatlog']
    resources['ocr_collection'] = client['invoicereviewgpt']['parsed_invoices']
    resources['vision_model'] = ChatOpenAI(model='gpt-4o',temperature=0.001,model_kwargs={"response_format":{"type":"json_object"}})
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

def get_vision_cost(prompt_tokens:int, completion_tokens:int) -> float:
    return round(((prompt_tokens/1000)*0.005)+((completion_tokens/1000)*0.015),5)

def get_model_name() -> str:
    return resources['model'].deployment_name.replace('-dev','')

def get_vision_model_name() -> str:
    return "gpt-4o"

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

    if chat_request.query is None or chat_request.query.strip() == '' or chat_request.session_id is None:
        raise HTTPException(status_code=400, detail="Invalid Request. Params 'query', and 'session_id' should not be Empty.")
    
    conversation_id: str = str(uuid.uuid4())
    config: dict = {"metadata": {"conversation_id": conversation_id}}
    start: time = time()

    with get_openai_callback() as cb:
        try:
            response:dict = resources['classifcation_chain'].invoke({"query":chat_request.query},config=config)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to Classify Query: {str(e)}")

        if response['class_label'] == "query" and response['subclass_label'] != "greetings":
            try:
                query_response = resources['response_chain'].invoke({"query":chat_request.query,
                                                                    "session_id":chat_request.session_id,
                                                                    "chat_format_instructions":query_response_parser.get_format_instructions()
                                                                    },config=config)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unable to Generate Response: {str(e)}")
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
    
    try:
        response = parse_obj_as(ChatResponse, response)
        resources['chatlog_collection'].insert_one(response.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to Insert Response: {str(e)}")
    return response

@app.post("/v1/customerservice-dev/")
async def chat_completion_dev(chat_request: ChatRequest) -> ChatResponse:
    response:dict = {
            'query':chat_request.query,
            'answer':'The generated answer will appear here',
            'class_label':'query',
            'session_id':chat_request.session_id,
            'conversation_id':chat_request.session_id+'_conv_id',
            'subclass_label':'product_query',
            'followup_questions':['New development on Product A.', 'Update contact information.', 'Share more details on credits.'],
            'context_id':['product_info.docx','training_guide.pptx','transition_guide.pdf'],
            'search_query':'Search query of '+chat_request.query+'. I really want it in two lines?',
            'hallucination_flag':0,
            'total_time':random.randint(100,300)/100,
            'prompt_tokens':random.randint(1000,2000),
            'completion_tokens':random.randint(100,200),
            'total_tokens':random.randint(1100,2200),
            'total_cost':random.randint(1,10)/1000,
            'model_name':'gpt-4o',
            'timestamp': datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d  %I:%M:%S %p"),
        }

    try:
        response = parse_obj_as(ChatResponse, response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to Parse Resposne: {str(e)}")
    return response

@app.get("/v1/analytics")
async def get_analytics() -> AnalyticsResponse:
    chatlog = resources["chatlog_collection"]
    try:
        start = time()
        response = {
                        'total_cost': round(list(chatlog.aggregate([{"$group": {"_id": None,"total_cost_sum": {"$sum": "$total_cost"}}}]))[0]['total_cost_sum'],3),
                        'total_requests': chatlog.count_documents({}),
                        'average_tokens_per_question': round(list(chatlog.aggregate([{"$group": {"_id": None,"average_tokens": {"$avg": "$total_tokens"}}}]))[0]['average_tokens']),
                        'average_response_time': round(list(chatlog.aggregate([{"$group": {"_id": None,"average_response_time": {"$avg": "$total_time"}}}]))[0]['average_response_time'],2),
                        'customer_questions': json.loads(pd.DataFrame([{'Class': i["_id"]["class_label"].replace('_', ' ').title(), 'Sub-Class': i["_id"]["subclass_label"].replace('_', ' ').title(), 'Count': i["count"]} for i in list(chatlog.aggregate([{"$group": {"_id": {"class_label": "$class_label", "subclass_label": "$subclass_label"}, "count": {"$sum": 1}}}]))]).to_json(orient='records')),
                        'tickets_orders': json.loads(pd.DataFrame([(key, value) for key, value in (list(chatlog.aggregate([{"$group": {"_id": None, "Ticket Queries": {"$sum": {"$cond": {"if": {"$eq": ["$class_label", "complaint"]}, "then": 1, "else": 0}}}, "Order Queries": {"$sum": {"$cond": {"if": {"$eq": ["$class_label", "order"]}, "then": 1, "else": 0}}}}}]))[0]).items() if key!='_id'],columns=['Category', 'Count']).to_json(orient='records')),
                        'periodic_traffic': json.loads(pd.DataFrame(list(chatlog.aggregate([{"$group": {"_id": "$timestamp", "Count": {"$sum": 1}}}, {"$sort": {"_id": 1}}]))).assign(Date=lambda df: pd.to_datetime(df['_id'], format='%Y-%m-%d %H:%M:%S %p', errors='coerce').dt.strftime('%Y-%m-%d')).groupby('Date', as_index=False)['Count'].sum().to_json(orient='records')),
                        'hourly_traffic': json.loads((pd.DataFrame(list(chatlog.aggregate([{"$project": {"hour": {"$substr": ["$timestamp", 11, 2]}}}, {"$group": {"_id": "$hour", "count": {"$sum": 1}}}, {"$sort": {"_id": 1}},{"$project": {"Hour": "$_id", "Count": "$count", "_id": 0}}]))).merge(pd.DataFrame({'Hour': [str(i).zfill(2) for i in range(24)]}), on='Hour', how='outer').fillna(0).astype({'Count': int})).assign(Hour=lambda df: pd.to_datetime(df['Hour'], format='%H').dt.strftime('%I %p')).to_json(orient='records')),
                        'hallucination_check': json.loads((pd.DataFrame(list(chatlog.aggregate([{"$group": {"_id": "$hallucination_flag", "count": {"$sum": 1}}},{"$project": {"Status": "$_id", "Count": "$count", "_id": 0}}]))).assign(Status=lambda x: x['Status'].map({'0': 'False', '1': 'True', None:'Not Invoked'}))).to_json(orient='records')),
                        'top_n_docs': json.loads(pd.DataFrame(sorted(list(chatlog.aggregate([{"$unwind": "$context_id"}, {"$group": {"_id": "$context_id", "Count": {"$sum": 1}}}, {"$project": {"_id": 0, "Document": "$_id", "Count": 1}}])),key=lambda d: d['Count'],reverse=True)[:10],columns=['Document','Count']).to_json(orient='records')),
                        'mean_response_time': json.loads(pd.DataFrame(list(chatlog.aggregate([{"$group": {"_id": "$class_label", "average_total_time": {"$avg": "$total_time"}}}, {"$project": {"_id": 0, "Category": "$_id", "Mean Response Time (in secs)": {"$round": ["$average_total_time", 2]}}}, {"$sort": {"average_total_time": 1}}])),columns=['Category','Mean Response Time (in secs)']).to_json(orient='records')),
                        'questions_per_dollar': json.loads(pd.DataFrame(list(chatlog.aggregate([{"$group": {"_id": "$class_label", "average_total_cost": {"$avg": "$total_cost"}}}, {"$project": {"_id": 0, "Category": "$_id", "average_total_cost": 1, "Mean Queries / $1": {"$toInt": {"$round": [{"$divide": [1, "$average_total_cost"]}]}}}}, {"$sort": {"average_total_cost": 1}}])), columns=['Category', 'Mean Queries / $1']).to_json(orient='records')),
                        'timestamp': datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d  %I:%M:%S %p")
                    }
        end = time()
        response.update({'total_time':round(end-start,2)})
    except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to Fetch Stats: {str(e)}")
    
    try:
        response = parse_obj_as(AnalyticsResponse, response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to Parse Stats Resposne: {str(e)}")

@app.post("/v1/ocr/")
async def invoice_OCR(invoice: UploadFile) -> OcrResponse:
    if invoice is None or (invoice.content_type!='image/jpeg' and invoice.content_type!='image/png'):
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPEG and PNG files are allowed.")
    try:
        encoded_image: bytes = base64.b64encode(invoice.file.read()).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during file encoding: {str(e)}")
    
    try:
        start: time = time()
        with get_openai_callback() as cb:
            response = resources['vision_model'].invoke([SystemMessage(content=[{"type":"text","text":"Convert all the elements in invoice to JSON. Return JSON only."}]),
                                                     HumanMessage(content=[{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{encoded_image}",}}])])
            response = response.dict()
            end = time()
            print(type(response))
        response.update({
                        'filename':invoice.filename,
                        'parsed_content':json.loads(response['content']),
                        'total_time':round(end-start,2),
                        'prompt_tokens':cb.prompt_tokens,
                        'completion_tokens':cb.completion_tokens,
                        'total_tokens': cb.total_tokens,
                        'total_cost': get_vision_cost(prompt_tokens=cb.prompt_tokens, completion_tokens=cb.completion_tokens),
                        'model_name':get_vision_model_name(),
                        'timestamp':datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S %p"),
                        })
        try:
            response = parse_obj_as(OcrResponse, response)
            resources['ocr_collection'].insert_one(response.__dict__)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to Insert Response: {str(e)}")
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during API request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)