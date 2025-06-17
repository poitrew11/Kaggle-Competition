import requests
from langchain_openai import ChatOpenAI
params = {"q": "funny cats"}
response = requests.get("http", params = params)

print(response.text)