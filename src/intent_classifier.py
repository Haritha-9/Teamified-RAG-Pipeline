from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import OPENAI_API_KEY

def classify_intent(query: str):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a query intent classifier.

Classify the user query into ONE of:
- fact_lookup
- date_question
- person_question
- summary
- unknown

Respond with only the category name.
"""),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"question": query})