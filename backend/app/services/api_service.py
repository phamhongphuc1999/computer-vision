from sanic import json


def ok_json(data):
    return json(data)


def bad_request_json(error):
    return json({"error": error}, status=400)
