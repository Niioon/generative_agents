from utils import create_agent_from_config
from pydantic.v1.tools import parse_file_as
from gen_agent import GenerativeAgent
import os
import pickle
def main():

    with open('api_token.txt', 'r') as file:
        api_key = file.read()
    os.environ['OPENAI_API_KEY'] = api_key

    path_1 = "characters/brian.txt"
    agent_1 = create_agent_from_config(path_1)
    path = 'saved_characters/test_agent'
    print(agent_1.generate_dialogue_response("Hi, I am Nion and I like Curry?"))
    print(agent_1.long_term_memory.memory_retriever.memory_stream)
    print(list(agent_1.long_term_memory.memory_retriever.vectorstore.docstore._dict.values()))

    docs_and_scores = agent_1.long_term_memory.memory_retriever.vectorstore.similarity_search_with_relevance_scores('nion')
    #print(docs_and_scores)

    GenerativeAgent.save_instance(agent_1, path)
    agent_loaded = GenerativeAgent.load_instance(path)
    print(agent_loaded.long_term_memory.memory_retriever.memory_stream)
    print(agent_loaded.generate_dialogue_response("What do you know about Nion?"))


if __name__ == '__main__':
    main()