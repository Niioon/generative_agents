import logging
import re
import math
from datetime import datetime
from typing import Any, Dict, List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import mock_now
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore

import json
import faiss
import os
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler('ltm.log', mode='w')
logger.addHandler(fh)


class GenerativeAgentMemory(BaseMemory):
    """Memory for the generative agent.
    Based on the code from:
    https://api.python.langchain.com/en/latest/_modules/langchain_experimental
    /generative_agents/memory.html#GenerativeAgentMemory"""
    llm_string: str
    """ Identifier of OPENAI model, used to create llm instance, used for recreation of llm when loading"""
    llm: Optional[BaseLanguageModel]
    """The underlying language model."""
    memory_retriever: Optional[TimeWeightedVectorStoreRetriever]
    """The retriever to fetch related memories, if not given fresh one will be created"""
    verbose: bool = False
    reflection_threshold: Optional[float] = None
    """When aggregate_importance exceeds reflection_threshold, stop to reflect."""
    current_plan: List[str] = []
    """The current plan of the agent."""
    # A weight of 0.15 makes this less important than it
    # would be otherwise, relative to salience and time
    importance_weight: float = 0.15
    """How much weight to assign the memory importance."""
    aggregate_importance: float = 0.0  # : :meta private:
    """Track the sum of the 'importance' of recent memories.

    Triggers reflection when it reaches reflection_threshold."""

    max_tokens_limit: int = 1200  # : :meta private:
    # input keys
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"
    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"
    reflecting: bool = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.llm:
            self.llm = ChatOpenAI(model_name=self.llm_string)
        if not self.memory_retriever:
            self.memory_retriever = self.create_new_memory_retriever()

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def _get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        """Return the 3 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given only the information above, what are the 3 most salient "
            "high-level questions we can answer about the subjects in the statements?\n"
            "Provide each question on a new line."
        )
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]
        )
        result = self.chain(prompt).invoke({'observations': observation_str})['text']
        return self._parse_list(result)

    def _get_insights_on_topic(
            self, topic: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            "Statements relevant to: '{topic}'\n"
            "---\n"
            "{related_statements}\n"
            "---\n"
            "What 5 high-level novel insights can you infer from the above statements "
            "that are relevant for answering the following question?\n"
            "Do not include any insights that are not relevant to the question.\n"
            "Do not repeat any insights that have already been made.\n\n"
            "Question: {topic}\n\n"
            "(example format: insight (because of 1, 5, 3))\n"
        )

        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join(
            [
                self._format_memory_detail(memory, prefix=f"{i + 1}. ")
                for i, memory in enumerate(related_memories)
            ]
        )
        result = self.chain(prompt).invoke(
            {'topic': topic,
             'related_statements': related_statements}
        )['text']
        # TODO: Parse the connections between memories and insights
        return self._parse_list(result)

    def pause_to_reflect(self, now: Optional[datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("Character is reflecting")
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(topic, now=now)
            for insight in insights:
                self.add_memory(insight, now=now)
            new_insights.extend(insights)
        return new_insights

    def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        score = self.chain(prompt).invoke({'memory_content': memory_content})['text']
        if self.verbose:
            logging.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def _score_memories_importance(self, memory_content: str) -> List[float]:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Always answer with only a list of numbers."
            + " If just given one memory still respond in a list."
            + " Memories are separated by semi colans (;)"
            + "\Memories: {memory_content}"
            + "\nRating: "
        )
        scores = self.chain(prompt).invoke({'memory_content': memory_content})['text']

        if self.verbose:
            logger.info(f"Importance scores: {scores}")
            print(f"Importance scores: {scores}")

        # Split into list of strings and convert to floats
        scores_list = [float(x) for x in scores.split(";")]

        return scores_list

    def add_memories(
            self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observations or memories to the agent's memory."""
        importance_scores = self._score_memories_importance(memory_content)

        self.aggregate_importance += max(importance_scores)
        memory_list = memory_content.split(";")
        documents = []

        for i in range(len(memory_list)):
            documents.append(
                Document(
                    page_content=memory_list[i],
                    metadata={"importance": importance_scores[i]},
                )
            )

        result = self.memory_retriever.add_documents(documents, current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
                self.reflection_threshold is not None
                and self.aggregate_importance > self.reflection_threshold
                and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def add_memory(
            self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        importance_score = self._score_memory_importance(memory_content)
        self.aggregate_importance += importance_score
        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}
        )
        result = self.memory_retriever.add_documents([document], current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
                self.reflection_threshold is not None
                and self.aggregate_importance > self.reflection_threshold
                and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def fetch_memories(
            self, observation: str, now: Optional[datetime] = None
    ) -> List[Document]:
        """Fetch related memories."""
        if now is not None:
            with mock_now(now):
                return self.memory_retriever.get_relevant_documents(observation)
        else:
            return self.memory_retriever.get_relevant_documents(observation)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}[{created_time}] {memory.page_content.strip()}"

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc)
        return self.format_memories_simple(result)

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return [self.relevant_memories_key, self.relevant_memories_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """This is called everytime before a chain of the generative agent is exucted.
        Returns key value pairs of variables which can be accessed in the prompt template"""
        queries = inputs.get(self.queries_key)
        now = inputs.get(self.now_key)
        if queries is not None:
            if self.verbose:
                logger.info("Fetching relevant memories for given queries")
                logger.info(f'queries: {queries}')
            relevant_memories = [
                mem for query in queries for mem in self.fetch_memories(query, now=now)
            ]
            return {
                self.relevant_memories_key: self.format_memories_detail(
                    relevant_memories
                ),
                self.relevant_memories_simple_key: self.format_memories_simple(
                    relevant_memories
                ),
            }

        most_recent_memories_token = inputs.get(self.most_recent_memories_token_key)
        if most_recent_memories_token is not None:
            return {
                self.most_recent_memories_key: self._get_memories_until_limit(
                    most_recent_memories_token
                )
            }
        return {}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this model run to memory."""
        # TODO: fix the save memory key
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if self.verbose:
            logging.info(f"inputs{inputs}")
            logging.info(f"outputs{outputs}")
        if mem:
            self.add_memory(mem, now=now)

    def clear(self) -> None:
        """Clear memory contents."""
        pass

    @staticmethod
    def save_instance(memory, path):

        memory_json = memory.json(exclude={'llm', 'memory_retriever'})

        with open( path + '/ltm_dict.json', 'w') as outfile:
            outfile.write(memory_json)

        # save vector store
        memory.memory_retriever.vectorstore.save_local(os.path.join(path, 'faiss_vector_store.index'))
        # separately save memory stream because langchain is stupid
        docs = memory.memory_retriever.memory_stream

    @staticmethod
    def load_instance(path: str):
        with open(path + '/ltm_dict.json', 'r') as openfile:
            ltm_dict = json.load(openfile)
        ltm_dict['llm'] = ChatOpenAI(model_name=ltm_dict['llm_string'])
        # load index
        vectorstore = FAISS.load_local(os.path.join(path, 'faiss_vector_store.index'), OpenAIEmbeddings())
        memory_retriever = GenerativeAgentMemory.create_new_memory_retriever(vectorstore)
        ltm_dict['memory_retriever'] = memory_retriever

        # because langchain is very very stupid, need to manually recreate the memory stream of the retriever
        # from the stored documents of the vectorstore
        docs = list(vectorstore.docstore._dict.values())
        # sort by buffer idx to ensure correct order
        docs.sort(key=lambda x: x.metadata['buffer_idx'], reverse=False)
        memory_retriever.memory_stream = docs

        return GenerativeAgentMemory(**ltm_dict)

    @staticmethod
    def create_new_memory_retriever(vectorstore=None):
        """Create a new vector store retriever unique to the agent."""
        if not vectorstore:
            embeddings_model = OpenAIEmbeddings()
            # Initialize the vectorstore as empty
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)

            def relevance_score_fn(score: float) -> float:
                """Return a similarity score on a scale [0, 1]."""
                return 1.0 - score / math.sqrt(2)

            vectorstore = FAISS(
                embeddings_model,
                index,
                InMemoryDocstore({}),
                {},
                relevance_score_fn=relevance_score_fn,
            )
        return TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, other_score_keys=["importance"], k=5
        )