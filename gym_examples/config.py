import os

from omegaconf import DictConfig
from omegaconf import OmegaConf


class ConfigSingleton:
    _config = None

    @staticmethod
    def load_config(config: DictConfig):
        override_config_path = "/app/override_config.yaml"
        if os.path.exists(override_config_path):
            override_cfg = OmegaConf.load(override_config_path)
            config = OmegaConf.merge(config, override_cfg)
        config.carla.host = os.environ.get("CARLA_CONTAINER", config.carla.host)
        ConfigSingleton._config = config

    @staticmethod
    def get_config():
        if ConfigSingleton._config is None:
            raise ValueError("Configuration has not been loaded")
        return ConfigSingleton._config


class ConfigProxy:
    @staticmethod
    def __getattr__(attr):
        config = ConfigSingleton.get_config()
        return getattr(config, attr)

    def to_dict(self):
        config = ConfigSingleton.get_config()
        return OmegaConf.to_container(config)

    def to_dictconfig(self):
        config = ConfigSingleton.get_config()
        return OmegaConf.create(config)


cfg = OmegaConf.load("/home/ekin/Scenic/gym_examples/default_config.yaml")
