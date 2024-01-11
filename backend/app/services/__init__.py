from sanic.request import Request
from sanic.response import HTTPResponse
from termcolor import colored


async def after_request(request: Request, response: HTTPResponse) -> HTTPResponse:
    try:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "deny"
    finally:
        return response


def log(message: str, keyword: str = "WARN"):
    if keyword == "WARN":
        print(colored("[WARN]", "yellow"), message)
    elif keyword == "ERROR":
        print(colored("[ERROR] " + message, "red"))
    elif keyword == "INFO":
        print(colored("[INFO]", "blue"), message)
    else:
        print(colored("[{}]".format(keyword), "cyan"), message)
