import json
import glob
from datetime import datetime

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

raw_documents = []

pr = set()
title = {}
for filename in glob.glob("data/github/*/*/issues-*.json"):
    repo = "/".join(filename.split("/")[2:4])

    print(f"Reading {filename}")
    with open(filename) as file:
        for item in json.load(file):
            issue_number = item["number"]
            body = item["body"]
            if item.get("pull_request", None) is None:
                if body is not None and len(body) > 0:
                    raw_documents.append(
                        Document(
                            page_content=body.replace("\r\n", "\n"),
                            metadata={
                                "url": f"https://github.com/{repo}/issues/{issue_number}",
                                "timestamp": datetime.fromisoformat(
                                    item["updated_at"]
                                ).isoformat(),
                                "type": "github-issue",
                                "github-repo": repo,
                                "github-issue-number": issue_number,
                                "github-issue-title": item["title"],
                            },
                        )
                    )
                title[repo, issue_number] = item["title"]

            else:
                pr.add((repo, issue_number))

    print(f"        {len(raw_documents)} documents have been loaded")

for filename in glob.glob("data/github/*/*/issue-comments-*.json"):
    repo = "/".join(filename.split("/")[2:4])

    print(f"Reading {filename}")
    with open(filename) as file:
        for item in json.load(file):
            issue_number = item["issue"]
            comment_number = item["id"]
            body = item["body"]
            if (repo, issue_number) not in pr and body is not None and len(body) > 0:
                raw_documents.append(
                    Document(
                        page_content=body.replace("\r\n", "\n"),
                        metadata={
                            "url": f"https://github.com/{repo}/issues/{issue_number}#issuecomment-{comment_number}",
                            "timestamp": datetime.fromisoformat(
                                item["updated_at"]
                            ).isoformat(),
                            "type": "github-issue-comment",
                            "github-repo": repo,
                            "github-issue-number": issue_number,
                            "github-issue-title": title[repo, issue_number],
                            "github-issue-comment-number": comment_number,
                        },
                    )
                )

    print(f"        {len(raw_documents)} documents have been loaded")

for filename in glob.glob("data/github/*/*/discussions-*.json"):
    repo = "/".join(filename.split("/")[2:4])

    print(f"Reading {filename}")
    with open(filename) as file:
        for discussion in json.load(file):
            body = discussion["body"]
            if (
                body is not None
                and len(body) > 0
                and "This discussion was created from the release" not in body
            ):
                raw_documents.append(
                    Document(
                        page_content=body.replace("\r\n", "\n"),
                        metadata={
                            "url": f"https://github.com/{repo}/discussions/{discussion['number']}",
                            "timestamp": datetime.fromisoformat(
                                discussion["createdAt"]
                            ).isoformat(),
                            "type": "github-discussion",
                            "github-repo": repo,
                            "github-discussion-number": discussion["number"],
                            "github-discussion-title": discussion["title"],
                        },
                    )
                )

            for comment in discussion["comments"]["nodes"]:
                body = comment["body"]
                if body is not None and len(body) > 0:
                    raw_documents.append(
                        Document(
                            page_content=body.replace("\r\n", "\n"),
                            metadata={
                                "url": f"https://github.com/{repo}/discussions/{discussion['number']}#discussioncomment-{comment['number']}",
                                "timestamp": datetime.fromisoformat(
                                    comment["createdAt"]
                                ).isoformat(),
                                "type": "github-discussion-comment",
                                "github-repo": repo,
                                "github-discussion-number": discussion["number"],
                                "github-discussion-title": discussion["title"],
                                "github-discussion-comment-number": comment["number"],
                            },
                        )
                    )

                for reply in comment["replies"]["nodes"]:
                    body = reply["body"]
                    if body is not None and len(body) > 0:
                        raw_documents.append(
                            Document(
                                page_content=body.replace("\r\n", "\n"),
                                metadata={
                                    "url": f"https://github.com/{repo}/discussions/{discussion['number']}#discussioncomment-{reply['number']}",
                                    "timestamp": datetime.fromisoformat(
                                        reply["createdAt"]
                                    ).isoformat(),
                                    "type": "github-discussion-comment-reply",
                                    "github-repo": repo,
                                    "github-discussion-number": discussion["number"],
                                    "github-discussion-title": discussion["title"],
                                    "github-discussion-comment-number": comment[
                                        "number"
                                    ],
                                    "github-discussion-reply-number": reply["number"],
                                },
                            )
                        )

    print(f"        {len(raw_documents)} documents have been loaded")

print("Done loading documents into memory")

documents = RecursiveCharacterTextSplitter(
    ["\n```", "\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=200
).split_documents(raw_documents)

print(f"{len(raw_documents)} have been split into {len(documents)} nodes")

embedding = OpenAIEmbeddings()

print(f"Loaded OpenAIEmbeddings")

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="hep-help",
    persist_directory="./hep-help-db",
)

print(f"Filled database with documents")

vectordb.persist()

print(f"Saved database to disk")
