from .product_config import product_config
from .objects import ClassificationLabels, QueryResponse, QueryRewriter
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder

query_classification_parser = JsonOutputParser(pydantic_object=ClassificationLabels)
query_rewriter_parser = JsonOutputParser(pydantic_object=QueryRewriter)
query_response_parser = JsonOutputParser(pydantic_object=QueryResponse)

query_classification_prompt = PromptTemplate(
    template=(
    f"You're a classification bot trained to redirect {product_config.org_name}'s customer questions. "
    f"{product_config.about}"
    f"{product_config.catalog}"
    "Classify the customer query into one of the following 4 classes and further into their "
    "subclasses if possible so that it can be answered by the correct department: \n\n"

    f"* 'query': If the customer is greeting or asking general questions about {product_config.org_name} or "
    f"about details like {product_config.topics['query']}, etc. \n"
    "Subclasses: greetings, product_query, company_related_query, and others. \n\n"

    f"* 'complaint': If the customer has any issues with orders like {product_config.topics['complaint']}, etc., that need human attention. \n"
    "Subclasses: create_ticket, view_ticket, update_ticket, cancel_ticket and others. \n\n"

    f"* 'order': If the customer wants to perform order related operations like {product_config.topics['order']}, etc.\n"
    "Subclasses: create_order, view_order, update_order, cancel_order, and others. \n\n"

    f"* 'tangential': If the customer asks questions unrelated to {product_config.org_name} or its products. \n"
    "Subclasses: unrelated, misleading, and others. \n\n"

    "Return answer only in JSON, Output Format: {format_instructions}\n"
    "Query: {query} \n"
    ),
    input_variables=['query'],
    partial_variables={"format_instructions": query_classification_parser.get_format_instructions()},
)

query_rewriter_prompt=PromptTemplate(
    template=(
    "Modify the latest Human question if it is a follow up question for better relevancy in search retrieval. If it is a greeting or a new topic (not a follow up question) then rewrite it in a better way to make it a standalone question using the conversation history.\n" 
    "For example: \n"

    "\n======\n"
    "Example 1 (For a Follow-up question in same topic) - If the follow-up question does not have the poduct name, include the product name as it helps for better retrieval: \n"
    "Human: Tell me about ChatGPT?\n"
    "AI: ChatGPT is an AI language model by OpenAI, adept at engaging in human-like conversations and assisting with a variety of tasks.\n"
    "Human: what is the price?\n\n"

    "search_query: What is the price of ChatGPT?\n\n"

    "Example 2 (For a question in a new topic) - No need to include the previous product name as the question is about a new topic: \n"
    "Human: Tell me about Apple iPhone\n"
    "AI: The iPhone is a popular line of smartphones designed and marketed by Apple Inc., known for its sleek design, powerful performance, and user-friendly interface.\n"
    "Human: What sizes do t-shirt come in?\n\n"

    "search_query: What are the common sizes available for t-shirts?\n"
    
    "\n======\n"
    "Return answer only in JSON, Output Format: {format_instructions}" 
    "\n======\n"
    "Current Conversation: \n"
    "{memory}\n"
    "Human: {query}\n"),
    input_variables=['memory','query'],
    partial_variables={"format_instructions": query_rewriter_parser.get_format_instructions()}
)

system_prompt_template = SystemMessagePromptTemplate.from_template(
    template=
    ("You are given the following extracted parts of marketing documents and a question. "
    "Read the following documents carefully. \n"

    "\n=========\n"
    "Relevant Sources: \n"
    "\n{context}"
    "\n=========\n"

    f"You are an expert customer support agent at {product_config.org_name} with excellent attention to detail. "
    "You should ONLY use the information in 'Relevant Sources' section provided above while answering. "
    "DON'T use your prior knowledge to answer customer question. "
    "Always provide a short conversational answer with maximum clarity. "
    "When it comes to product related query I always want you to double check you're answering with correct product name and details. \n"
    "In a follow up question, answer to the same product customer is talking about."
    "If you don't know the answer or if sufficient details are not present in the relevant documents, "
    "DON'T try to make up an answer, it will create legal complications for misdirecting customers, just say you don't know and redirect the customer to other agents available through phone "
    f"at {product_config.contact_info['phone_number']}. The agents are available from {product_config.contact_info['days']} {product_config.contact_info['time']} ({product_config.contact_info['time_zone']}). "
    "Also, generate EXACTLY 3 followup questions that can be answered from the content. \n"

    "\n=========\n"
    "Return answer only in JSON, Output Format: {chat_format_instructions}"
    "\n=========\n"),
    input_variables=['context','query'],
    partial_variables={"chat_format_instructions": query_response_parser.get_format_instructions()}
)

human_prompt_template = HumanMessagePromptTemplate.from_template(template = '{query}\n', input_variables = ['query'])
query_response_prompt = ChatPromptTemplate.from_messages([system_prompt_template,
                                                  MessagesPlaceholder(variable_name="memory"),
                                                  human_prompt_template])