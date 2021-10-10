import requests

url = "http://localhost:7000/predict"
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 10}
response = requests.post(url, json=customer).json()

print(response)
