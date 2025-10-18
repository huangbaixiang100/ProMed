import requests
import json


class InputLengthError(requests.RequestException):
    """The length of input exceeds the max length"""


class InvalidKeyError(requests .RequestException):
    """The key is invalid."""


def call_gpt(client,model,messages):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
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
            return f"{model} error: {error_message}"
        else:
            return f"{model} error: Unexpected error"
    except Exception as e:
        return f"{model} error: {str(e)}"