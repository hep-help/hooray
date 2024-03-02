import re
import sys
import logging
import math
from datetime import datetime

from langchain.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

def format_metadata(metadata):
    m = metadata
    suffix = f" ({datetime.fromisoformat(m['timestamp']).strftime('%d %B, %Y').lstrip('0')})"
    if metadata["type"] == "github-issue":
        return f"GitHub issue [{m['github-repo']}#{m['github-issue-number']}]({m['url']}), {m['github-issue-title']}{suffix}"
    elif metadata["type"] == "github-issue-comment":
        return f"GitHub issue [{m['github-repo']}#{m['github-issue-number']} (comment)]({m['url']}), {m['github-issue-title']}{suffix}"
    elif metadata["type"] == "github-discussion":
        return f"GitHub discussion [{m['github-repo']}#{m['github-discussion-number']}]({m['url']}), {m['github-discussion-title']}{suffix}"
    elif metadata["type"] == "github-discussion-comment":
        return f"GitHub discussion [{m['github-repo']}#{m['github-discussion-number']} (comment)]({m['url']}), {m['github-discussion-title']}{suffix}"
    elif metadata["type"] == "github-discussion-comment-reply":
        return f"GitHub discussion [{m['github-repo']}#{m['github-discussion-number']} (reply)]({m['url']}), {m['github-discussion-title']}{suffix}"
    else:
        return f"[{m['url']}]({m['url']})" + suffix


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

        matching_urls = set()
        matching_documents = []
        logger.info("Getting 5 documents by direct similarity test")
        for document in basic_retriever.get_relevant_documents(query=query):
            if document.metadata["url"] not in matching_urls:
                matching_urls.add(document.metadata["url"])
                matching_documents.append(document)
        logger.info("Getting as many as 15 more by letting the LLM vary the query")
        for document in retriever.get_relevant_documents(query=query):
            if document.metadata["url"] not in matching_urls:
                matching_urls.add(document.metadata["url"])
                matching_documents.append(document)

        logger.info(f"Found a total of {len(matching_documents)} unique documents")

        logger.info("Asking an LLM to rank the documents by usefulness")
        ranking_llm = ChatOpenAI(temperature=0.5)

        numer = [0] * len(matching_documents)
        denom = [0] * len(matching_documents)
        for trial in range(1):
            response = ranking_llm.invoke(
                "\n\n".join(
                    f'<message number="{i + 1}">\n{x.page_content}\n</message>'
                    for i, x in enumerate(matching_documents)
                )
                + f"\n\n<question>{query}</question>\n\nInstructions: For each "
                'of the numbered messages in <message number="NUMBER"> tags, '
                "provide an integer score between 0 and 100 that characterizes "
                "how useful the message would be to me in finding the answer to the "
                "question in <question> tags. Return your responses as a "
                "space-delimited sequence of message NUMBER=SCORE pairs and nothing else."
            )
            logger.info(f"LLM response: {response.content}")

            ranking = [None] * len(matching_documents)
            for index, score in re.findall("([0-9]+)\s*=\s*([0-9]+)", response.content):
                good_index = int(index) - 1
                if 0 <= good_index < len(ranking):
                    ranking[good_index] = int(score)

            for i in range(len(matching_documents)):
                if ranking[i] is not None:
                    numer[i] += ranking[i]
                    denom[i] += 1

        final_ranking = [float("-inf")] * len(matching_documents)
        for i in range(len(matching_documents)):
            if denom[i] != 0:
                final_ranking[i] = round(numer[i] / denom[i])

        logger.info(f"final ranking: {final_ranking}")

        ranked_documents = list(zip(final_ranking, matching_documents))
        ranked_documents.sort(key=lambda pair: -pair[0])

        logger.info("Asking an LLM to summarize the usefulness of each document")
        summarizing_llm = ChatOpenAI(temperature=0.5)

        ranked_documents_with_summaries = []
        for score, document in ranked_documents:
            if math.isfinite(score):
                score = f"{score:.0f}"
                response = summarizing_llm.invoke(
                    f"<message>\n{document.page_content}\n</message>\n\n"
                    f"<question>\n{query}\n</question>\n\nInstructions: Given the "
                    f"message in <message> tags, explain whether this message will "
                    "help me find an answer to my question in <question> tags, "
                    "and if so, how I can use specific examples in that message. "
                    "Respond with only a few brief sentences, no more than 100 words."
                )
                logger.info(f"LLM response: {response.content}")
            else:
                score = "_(unranked)_"
                response = None

            ranked_documents_with_summaries.append(
                (score, document, "" if response is None else response.content)
            )

        logger.info(str(cb))

    final_document = "### Potentially useful sources\n\n" + "\n\n".join(
        f"**Score: {score}%:** {format_metadata(document.metadata)}\n\n{summary}"
        for score, document, summary in ranked_documents_with_summaries
    )
    print("\n\n------------------------------------------------\n\n")
    print(final_document)
    print("\n\n------------------------------------------------\n\n")
