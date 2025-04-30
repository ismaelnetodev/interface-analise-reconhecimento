import requests

url = "http://127.0.0.1:8000/gestao/api/estudantes"

response = requests.get(url)

if response.status_code == 200:
    estudantes = response.json()

    print("Estudantes:", estudantes)