from app import create_app, create_routes, AppConfig
from app.services.log_service import app_logger

sanic_app = create_app()
create_routes(
    sanic_app,
)
app_logger.info(f"environment: {AppConfig.env}")

if __name__ == "__main__":
    sanic_app.run(
        host=AppConfig.Global.HOST,
        port=AppConfig.Global.PORT,
        auto_reload=True,
        debug=False,
        access_log=True,
    )
