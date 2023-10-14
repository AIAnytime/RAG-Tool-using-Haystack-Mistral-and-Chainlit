import chainlit as cl
from datasets import load_dataset
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline
from haystack.utils import print_answers
import os
from dotenv import load_dotenv
load_dotenv()

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")

document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(dataset)

retriever = BM25Retriever(document_store=document_store, top_k=3)

prompt_template = PromptTemplate(
    prompt="""
    Answer the question truthfully based solely on the given documents. If the documents do not contain the answer to the question, say that answering is not possible given the available information. Your answer should be no longer than 50 words.
    Documents:{join(documents)}
    Question:{query}
    Answer:
    """,
    output_parser=AnswerParser(),
)

HF_TOKEN = os.environ.get("HF_TOKEN")

prompt_node = PromptNode(
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1", api_key=HF_TOKEN, default_prompt_template=prompt_template
)

generative_pipeline = Pipeline()
generative_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
generative_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

@cl.on_message
async def main(message: str):
    response = await cl.make_async(generative_pipeline.run)(message)
    sentences = response['answers'][0].answer.split('\n')

    # Check if the last sentence doesn't end with '.', '?', or '!'
    if sentences and not sentences[-1].strip().endswith(('.', '?', '!')):
        # Remove the last sentence
        sentences.pop()

    result = '\n'.join(sentences[1:])
    await cl.Message(author="Bot", content=result).send()

