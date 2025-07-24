from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks import StreamingStdOutCallbackHandler

callbacks = [StreamingStdOutCallbackHandler()]
llm = ChatOllama(model="llama3.1:latest")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Please answer the user's questions kindly. Answer me in Korean no matter what."),
    MessagesPlaceholder(variable_name='messsages1'),
])

chain = prompt | llm | StrOutputParser()