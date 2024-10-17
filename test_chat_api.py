import requests
import json

def test_completion():
    url = "http://localhost:5000/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0.7,
        "stop": ["."],
        "mode": "baseline"
    }

    response = requests.post(url, headers=headers, json=data)
    print("Completion API Response:")
    print(json.dumps(response.json(), indent=2))
    print("\n")

def test_chat_completion():
    url = "http://localhost:5000/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke about programming."}
        ],
        "max_tokens": 100,
        "temperature": 0.8,
        "stop": ["\n"],
        "mode": "baseline"
    }

    response = requests.post(url, headers=headers, json=data)
    print("Chat Completion API Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_completion()
    test_chat_completion()