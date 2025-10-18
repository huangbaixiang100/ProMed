import requests
import json


class InputLengthError(requests.RequestException):
    """The length of input exceeds the max length"""


class InvalidKeyError(requests .RequestException):
    """The key is invalid."""


def call_model(message,url=''):
    headers = {
        "Content-Type": "application/json"
    }
    if type(message) is str:
        messages=[{ "role": "user","content": message}]
    elif type(message) is list:
        messages=message
    else:
        return (f"error: type of message must be string or list")
    data = json.dumps({
        "model": "qwen2.5",
        "messages": messages
    })

    try:
        raw_response = requests.post(url, headers=headers, data=data)
        raw_response.raise_for_status()  # Raises stored HTTPError, if one occurred.

        res = json.loads(raw_response.text)
        words = res['choices'][0]['message']['content']
        #print(words)
        return words
    except requests.HTTPError as e:
        if e.response.status_code in (400, 401):
            # Handle specific error codes if needed
            error_info = e.response.json()
            error_message = error_info.get('error', {}).get('message', 'Unknown error')
            return f"error: {error_message}"
        else:
            return "error: Unexpected error"
    except Exception as e:
        return f"error: {str(e)}"


def call_gpt(client,model,messages,temperature=0.6):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False
        )
        words=response.choices[0].message.content
        print(words)
        return words
    except requests.HTTPError as e:
        if e.response.status_code in (400, 401):
            # Handle specific error codes if needed
            error_info = e.response.json()
            error_message = error_info.get('error', {}).get('message', 'Unknown error')
            return f"{model} !model error: {error_message}"
        else:
            return f"{model} !model error: Unexpected error"
    except Exception as e:
        return f"{model} !model error: {str(e)}"


