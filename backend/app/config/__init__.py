from app.config.global_config import GlobalConfig


class _AppConfig:
    def __init__(self):
        self.Global = GlobalConfig()
        self.env = None


AppConfig = _AppConfig()
