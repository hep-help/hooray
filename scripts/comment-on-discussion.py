import os
import time

import jwt
import requests

DISCUSSION_ID = os.environ["DISCUSSION_ID"]

APP_ID = "849365"
PRIVATE_KEY_PATH = "./key.pem"

def generate_jwt(app_id, private_key_path):
    # Current time and expiration time
    now = int(time.time())
    expiration_time = now + (10 * 60)
    payload = {
        'iat': now,
        'exp': expiration_time,
        'iss': app_id
    }
    # Read the private key
    with open(private_key_path, 'rb') as key_file:
        private_key = jwt.jwk_from_pem(key_file.read())
    instance = jwt.JWT()
    token = instance.encode(payload, private_key, alg='RS256')
    return token

def get_installation_token(jwt, installation_id):
    headers = {
        'Authorization': f'Bearer {jwt}',
        'Accept': 'application/vnd.github+json'
    }
    url = f'https://api.github.com/app/installations/{installation_id}/access_tokens'
    response = requests.post(url, headers=headers)
    response_data = response.json()
    return response_data.get('token')

jwt_token = generate_jwt(APP_ID, PRIVATE_KEY_PATH)
installation_id = "48061001"
installation_token = get_installation_token(jwt_token, installation_id)

headers = {
    "Authorization": f"bearer {installation_token}",
    "Content-Type": "application/json",
}

payload = {
    "query": """mutation {
    addDiscussionComment(
        input: {
            body: "This is a test."
            discussionId: "%s"
            clientMutationId: "hep-helper"
        }
    ) {
        clientMutationId comment { id body }
    }
}""".format(DISCUSSION_ID)
}

response = requests.post("https://api.github.com/graphql", json=payload)
