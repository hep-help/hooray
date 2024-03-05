import glob
import json
import os
from datetime import datetime

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

github_issues = {}

pr = set()
for filename in glob.glob("data/github/*/*/issues-*.json"):
    repo = "/".join(filename.split("/")[2:4])

    print(f"Reading {filename}")
    with open(filename) as file:
        for item in json.load(file):
            issue_number = item["number"]
            body = item["body"]
            if item.get("pull_request", None) is None:
                if body is None:
                    body = ""
                github_issues[repo, issue_number] = [
                    {
                        "page_content": body.replace("\r\n", "\n"),
                        "url": f"https://github.com/{repo}/issues/{issue_number}",
                        "timestamp": datetime.fromisoformat(item["updated_at"]).isoformat(),
                        "type": "github-issue",
                        "github-repo": repo,
                        "github-issue-number": issue_number,
                        "github-issue-title": item["title"],
                        "github-issue-comment-number": 0,
                    }
                ]
            else:
                pr.add((repo, issue_number))

    print(f"        {len(github_issues)} GitHub issues have been loaded")

for filename in glob.glob("data/github/*/*/issue-comments-*.json"):
    repo = "/".join(filename.split("/")[2:4])

    print(f"Reading {filename}")
    comment_count = 0
    with open(filename) as file:
        for item in json.load(file):
            issue_number = item["issue"]
            comment_number = item["id"]
            body = item["body"]
            if (repo, issue_number) not in pr and body is not None and len(body) > 0:
                assert (repo, issue_number) in github_issues
                github_issues[repo, issue_number].append(
                    {
                        "page_content": body.replace("\r\n", "\n"),
                        "url": f"https://github.com/{repo}/issues/{issue_number}#issuecomment-{comment_number}",
                        "timestamp": datetime.fromisoformat(item["updated_at"]).isoformat(),
                        "type": "github-issue-comment",
                        "github-issue-comment-number": comment_number,
                    }
                )
                comment_count += 1

    print(f"        {comment_count} GitHub issue comments have been loaded")

print("Sorting the comments on each issue by their comment numbers")
for github_issue in github_issues.values():
    github_issue.sort(key=lambda x: x["github-issue-comment-number"])

github_discussions = {}

for filename in glob.glob("data/github/*/*/discussions-*.json"):
    repo = "/".join(filename.split("/")[2:4])

    print(f"Reading {filename}")
    with open(filename) as file:
        for discussion in json.load(file):
            body = discussion["body"]
            if body is None or "This discussion was created from the release" in body:
                body = ""

            github_discussions[repo, discussion["number"]] = [
                {
                    "page_content": body.replace("\r\n", "\n"),
                    "url": f"https://github.com/{repo}/discussions/{discussion['number']}",
                    "timestamp": datetime.fromisoformat(discussion["createdAt"]).isoformat(),
                    "type": "github-discussion",
                    "github-repo": repo,
                    "github-discussion-number": discussion["number"],
                    "github-discussion-title": discussion["title"],
                }
            ]

            for comment in discussion["comments"]["nodes"]:
                body = comment["body"]
                if body is not None and len(body) > 0:
                    github_discussions[repo, discussion["number"]].append(
                        {
                            "page_content": body.replace("\r\n", "\n"),
                            "url": f"https://github.com/{repo}/discussions/{discussion['number']}#discussioncomment-{comment['number']}",
                            "timestamp": datetime.fromisoformat(comment["createdAt"]).isoformat(),
                            "type": "github-discussion-comment",
                            "github-discussion-comment-number": comment["number"],
                        }
                    )

                for reply in comment["replies"]["nodes"]:
                    body = reply["body"]
                    if body is not None and len(body) > 0:
                        github_discussions[repo, discussion["number"]].append(
                            {
                                "page_content": body.replace("\r\n", "\n"),
                                "url": f"https://github.com/{repo}/discussions/{discussion['number']}#discussioncomment-{reply['number']}",
                                "timestamp": datetime.fromisoformat(reply["createdAt"]).isoformat(),
                                "type": "github-discussion-comment-reply",
                                "github-discussion-reply-number": reply["number"],
                            }
                        )

    print(f"        {len(github_discussions)} GitHub discussions have been loaded, with their comments and replies")

print("Done loading documents into memory")

print("Formatting everything as whole-conversation Documents")
raw_documents = []
splitter = "\n\n--------------------------------------------------\n\n"

for github_issue in github_issues.values():
    metadata = dict(github_issue[0])
    del metadata["page_content"]
    del metadata["github-issue-comment-number"]
    raw_documents.append(
        Document(
            page_content=splitter.join(
                comment["page_content"] for comment in github_issue
            ),
            metadata=metadata
        )
    )

for github_discussion in github_discussions.values():
    metadata = dict(github_discussion[0])
    del metadata["page_content"]
    raw_documents.append(
        Document(
            page_content=splitter.join(
                comment["page_content"] for comment in github_discussion
            ),
            metadata=metadata
        )
    )

documents = RecursiveCharacterTextSplitter(
    [splitter, "\n```", "\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=200
).split_documents(raw_documents)

print(f"{len(raw_documents)} raw-documents have been split into {len(documents)} documents")

embedding = OpenAIEmbeddings()

print(f"Loaded OpenAIEmbeddings")

# FIXME!
documents = documents[:10]

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="hep-help",
    persist_directory="./hep-help-db",
)

print(f"Filled database with documents")

vectordb.persist()

print(f"Saved database to disk")

path = "./hep-help-db/raw-documents"
if not os.path.exists(path):
    os.mkdir(path)

for raw_document in raw_documents:
    if raw_document.metadata["type"] == "github-issue":
        typed_path = path + "/github-issue"
        if not os.path.exists(typed_path):
            os.mkdir(typed_path)

        org, repo = raw_document.metadata["github-repo"].split("/")
        org_path = typed_path + "/" + org
        if not os.path.exists(org_path):
            os.mkdir(org_path)
        repo_path = org_path + "/" + repo
        if not os.path.exists(repo_path):
            os.mkdir(repo_path)

        filename = repo_path + f"/{raw_document.metadata['github-issue-number']}.json"
        with open(filename, "w") as file:
            data = dict(raw_document.metadata)
            data["page_content"] = raw_document.page_content
            json.dump(data, file)


    elif raw_document.metadata["type"] == "github-discussion":
        typed_path = path + "/github-discussion"
        if not os.path.exists(typed_path):
            os.mkdir(typed_path)

        org, repo = raw_document.metadata["github-repo"].split("/")
        org_path = typed_path + "/" + org
        if not os.path.exists(org_path):
            os.mkdir(org_path)
        repo_path = org_path + "/" + repo
        if not os.path.exists(repo_path):
            os.mkdir(repo_path)

        filename = repo_path + f"/{raw_document.metadata['github-discussion-number']}.json"
        with open(filename, "w") as file:
            data = dict(raw_document.metadata)
            data["page_content"] = raw_document.page_content
            json.dump(data, file)

print(f"Saved raw-documents to disk")
