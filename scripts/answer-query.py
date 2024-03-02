import json
import logging
import math
import re
import sys
from datetime import datetime

import openai
from langchain_core.documents import Document
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI


def format_metadata(metadata):
    m = metadata
    prefix = f"**{datetime.fromisoformat(m['timestamp']).strftime('%d %B, %Y').lstrip('0')}:** "
    if metadata["type"] == "github-issue":
        return f"{prefix}GitHub issue [{m['github-repo']}#{m['github-issue-number']}]({m['url']}), {m['github-issue-title']}"
    elif metadata["type"] == "github-issue-comment":
        return f"{prefix}GitHub issue [{m['github-repo']}#{m['github-issue-number']} (comment)]({m['url']}), {m['github-issue-title']}"
    elif metadata["type"] == "github-discussion":
        return f"{prefix}GitHub discussion [{m['github-repo']}#{m['github-discussion-number']}]({m['url']}), {m['github-discussion-title']}"
    elif metadata["type"] == "github-discussion-comment":
        return f"{prefix}GitHub discussion [{m['github-repo']}#{m['github-discussion-number']} (comment)]({m['url']}), {m['github-discussion-title']}"
    elif metadata["type"] == "github-discussion-comment-reply":
        return f"{prefix}GitHub discussion [{m['github-repo']}#{m['github-discussion-number']} (reply)]({m['url']}), {m['github-discussion-title']}"
    else:
        return f"{prefix}[{m['url']}]({m['url']})"


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("langchain").setLevel(logging.INFO)

    logger = logging.getLogger("answer-query")
    logger.setLevel(logging.INFO)

    with get_openai_callback() as cb:
        logger.info("Loading database")
        embedding = OpenAIEmbeddings()
        vectordb = Chroma(
            collection_name="hep-help",
            persist_directory="./hep-help-db",
            embedding_function=embedding,
        )

        logger.info("Waiting for query")
        query = sys.stdin.read()
        logger.info(f"Query is {query!r}")

        basic_retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        multiquery_llm = ChatOpenAI(temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            retriever=basic_retriever, llm=multiquery_llm
        )

        unique_urls = set()
        unique_documents = []
        logger.info("Getting 5 documents by direct similarity test")
        for document in basic_retriever.get_relevant_documents(query=query):
            if document.metadata["url"] not in unique_urls:
                unique_urls.add(document.metadata["url"])
                unique_documents.append(document)
        logger.info("Getting as many as 15 more by letting the LLM vary the query")
        for document in retriever.get_relevant_documents(query=query):
            if document.metadata["url"] not in unique_urls:
                unique_urls.add(document.metadata["url"])
                unique_documents.append(document)

        logger.info(f"Found a total of {len(unique_documents)} unique documents")

        matching_documents = []
        for document in unique_documents:
            if document.metadata["type"] == "github-issue":
                m = document.metadata
                filename = f"./hep-help-db/raw-documents/{m['type']}/{m['github-repo']}/{m['github-issue-number']}.json"
            elif document.metadata["type"] == "github-discussion":
                m = document.metadata
                filename = f"./hep-help-db/raw-documents/{m['type']}/{m['github-repo']}/{m['github-discussion-number']}.json"
            else:
                continue

            with open(filename) as file:
                data = json.load(file)
                page_content = data.pop("page_content")
                matching_documents.append(Document(page_content=page_content, metadata=data))

        logger.info(f"Which correspond to {len(matching_documents)} raw documents")

        logger.info("Asking an LLM to rank the documents by usefulness")
        ranking_summarizing_llm = ChatOpenAI(temperature=0.5)

        ranked_documents_with_summaries = []
        for document in matching_documents:
            page_content = document.page_content

            while True:
                try:
                    response = ranking_summarizing_llm.invoke(
                        f"<conversation>\n{page_content}\n</conversation>\n\n"
                        f"<question>{query}</question>\n\nInstructions: Given the "
                        "conversation in <conversation> tags, determine whether any part "
                        "of it will help me find an answer to my question in <question> "
                        "tags, and if so, how I can use specific examples in that "
                        "conversation and what I should look for. Also, provide an "
                        "integer score between 0 and 100 that characterizes how well "
                        "the conversation addresses my question. The score should be "
                        "greater than 75 if it directly addresses my question, it should "
                        "be between 25 and 75 if it doesn't directly address my question "
                        "but contains information that I can use, and the score should "
                        "be below 25 if it doesn't contain anything useful. Return your "
                        "response in the format \"Score: NUMBER out of 100\", followed by a newline and a few brief "
                        "sentences (no more than 100 words) explaining how the conversation "
                        "is useful and what examples I should look for in the conversation "
                        "to help answer to my question."
                    )
                except openai.BadRequestError:
                    page_content = page_content[:len(page_content) // 2]
                    if len(page_content) < 5000:
                        response = None
                        break
                else:
                    break

            if response is None:
                continue

            logger.info(f"LLM response: {response.content}")

            score = float("-inf")
            score_match = re.search("([0-9]+)", response.content)
            if score_match is not None:
                score = int(score_match.group(1))

            if score >= 25:
                ranked_documents_with_summaries.append(
                    (score, document, response.content)
                )

        logger.info(str(cb))

    ranked_documents_with_summaries.sort(key=lambda pair: -pair[0])

    final_document = "### Potentially useful sources\n\n" + "\n\n".join(
        f"{format_metadata(document.metadata)}\n\n{summary}"
        for score, document, summary in ranked_documents_with_summaries
    )
    print("\n\n------------------------------------------------\n\n")
    print(final_document)
    print("\n\n------------------------------------------------\n\n")
