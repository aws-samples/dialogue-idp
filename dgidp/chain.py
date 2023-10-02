import json
import os
import pathlib
import pickle
from typing import Dict, List, Tuple

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import \
    SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS, Weaviate
from pydantic import BaseModel

class CustomChain(Chain, BaseModel):

    vstore: FAISS
    chain: BaseCombineDocumentsChain
    key_word_extractor: Chain

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs["question"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        if chat_history_str:
            new_question = self.key_word_extractor.run(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        print(new_question)
        docs = self.vstore.similarity_search(new_question, k=3)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = self.chain.combine_docs(docs, **new_inputs)

        ## Dedupe source list
        source_list = [doc.metadata['source'] for doc in docs]

        source_string = "\n\n*Sources:* "
        for i, source in enumerate(set(source_list)):
            source_string += f"<a href=\"https://{source}\" target=\"_blank\">[{i}]</a>"

        final_answer = answer + source_string
        return {"answer": final_answer}

def get_new_chain1(vectorstore, rephraser_llm, final_output_llm, isFlan) -> Chain:
    _eg_template = """## Example:
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question: {answer}"""
    _eg_prompt = PromptTemplate(
        template=_eg_template,
        input_variables=["chat_history", "question", "answer"],
    )

    _prefix = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. You should assume that the question is related to Hugging Face Code."""
    _suffix = """## Example:
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    #### LOAD VSTORE WITH REPHRASE EXAMPLES
    with open("rephrase_eg.pkl", 'rb') as f:
        rephrase_example_selector = pickle.load(f)

    prompt = FewShotPromptTemplate(
        prefix=_prefix,
        suffix=_suffix,
        example_selector=rephrase_example_selector,
        example_prompt=_eg_prompt,
        input_variables=["question", "chat_history"],
    )

    key_word_extractor = LLMChain(llm=rephraser_llm, prompt=prompt)

    EXAMPLE_PROMPT = PromptTemplate(
        template=">Example:\nContent:\n---------\n{page_content}\n----------\nSource: {source}",
        input_variables=["page_content", "source"],
    )

    flan_template = """
    {context}
    Based on the above documentation, answer the user's question in markdown: {question}"""

    PROMPT = PromptTemplate(template=flan_template, input_variables=["question", "context"])

    doc_chain = load_qa_chain(
        final_output_llm,
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT,
        verbose=True
    )
    return CustomChain(chain=doc_chain, vstore=vectorstore, key_word_extractor=key_word_extractor)


def _get_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = ""
    for human_s, ai_s in chat_history[-1:]:
        human = f"Human: " + human_s
        ai = f"Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])

    
    return buffer
