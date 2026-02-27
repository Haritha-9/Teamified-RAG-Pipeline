from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import OPENAI_API_KEY
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# model = ChatOpenAI()

def build_answer_chain():
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY
    )
    system_instruction = """You are a Philippine History assistant.
                        - Provide the final answer in ONLY 2 lines.- 
                        -If answers is not found provide answer is not available'
                        - Do not hallucinate."""
    prompt = ChatPromptTemplate.from_messages([("system", system_instruction),
                                           ("human", """Intent: {intent}
Context:{context}
Question:{question}
Answer:""") ])

    parser = StrOutputParser()

    chain = prompt | model| parser
    return chain