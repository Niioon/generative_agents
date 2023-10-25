import os
from datetime import datetime, timedelta
from typing import List
from termcolor import colored

import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from utils import relevance_score_fn, create_new_memory_retriever, interview_agent
from gen_agent import GenerativeAgent
from memory import GenerativeAgentMemory
from langchain.memory import ConversationSummaryMemory

def main():

    os.environ['OPENAI_API_KEY'] = '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'
    USER_NAME = "Nion"  # The name you want to use when interviewing the agent.
    LLM = ChatOpenAI(model_name="gpt-3.5-turbo")

    miriams_memory = GenerativeAgentMemory(
        llm=LLM,
        memory_retriever=create_new_memory_retriever(),
        verbose=True,
        reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
    )

    short_term = ConversationSummaryMemory(llm=OpenAI(temperature=0), input_key='observation')

    miriam = GenerativeAgent(
        name="Miriam",
        age=26,
        character_traits="extraverted, spirited, empathetic, educated",  # You can add more persistent character_traits here
        status="Finishing her Masters Degree in Psychology with a focus on education",
        memory_retriever=create_new_memory_retriever(),
        llm=LLM,
        short_term_memory= short_term,
        long_term_memory=miriams_memory,
        verbose=True
    )



    miriams_observations = [
        "Miriam lives in a shared flat with Svenja, Aylin, Nion, Max and Marcel. They frequently chat about their lives in the kitchen",
        "Miriam finfished her Master Thesis a month ago and is now enjoying some free time",
        "Miriam likes riding the bike and going for runs",
        "Miriam and Nion plan to talk about Nion's university project this evening",
        "Miriam has a a passion for sewing and frequently makes her own clothes",
        "Miriam is not a morning person and tends to sleep in late",
        "Miriam likes dad jokes",

    ]

    for observation in miriams_observations:
        miriam.long_term_memory.add_memory(observation)

    answer = interview_agent(miriam, "What are your plans for tonight?", USER_NAME)
    print(answer)


if __name__ == '__main__':
    main()
