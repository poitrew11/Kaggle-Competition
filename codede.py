import requests

params = {"q": "funny cats"}
response = requests.get("http", params = params)

print(response.text)