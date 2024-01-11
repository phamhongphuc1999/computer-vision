from sanic import Sanic
from sanic_cors import CORS

from app.config import AppConfig
from app.controller import global_blueprint
from app.services import log, after_request


def _register_extensions(sanic_app: Sanic):
    from app import extensions

    extensions.cors = CORS(sanic_app, resources={r"/*": {"origins": "*"}})


def _register_hooks(sanic_app: Sanic):
    sanic_app.register_middleware(after_request, "response")


def create_app(*config_cls) -> Sanic:
    log(
        message="Sanic application initialized with {}".format(", ".join([config.__name__ for config in config_cls])),
        keyword="INFO",
    )

    sanic_app = Sanic(name='my-hello-world-app')
    for _config in config_cls:
        sanic_app.config.update_config(_config)

    _register_extensions(sanic_app)
    _register_hooks(sanic_app)
    return sanic_app


def create_routes(sanic_app: Sanic, **kwargs):
    sanic_app.blueprint(global_blueprint)
    for key, value in kwargs.items():
        pass
