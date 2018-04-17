import os
import logging.config
import yaml


class Logger:
    def __init__(self):
        self.setup_logging(self)
        logger = logging.getLogger(__name__)
        logger.info('Hi, foo')

    @staticmethod
    def setup_logging(
        self, default_path='logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG'
    ):
        """Setup logging configuration
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

