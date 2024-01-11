from sanic import Blueprint, Request

from app.services.api_service import ok_json, bad_request_json

global_blueprint = Blueprint("global_blueprint", url_prefix="/")


@global_blueprint.get("/ping")
async def ping_server(request: Request):
    try:
        return ok_json({"status": "ok"})
    except Exception as error:
        return bad_request_json(str(error))
