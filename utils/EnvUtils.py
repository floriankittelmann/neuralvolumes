import os
import json


class EnvUtils:

    def get_env(self) -> dict:
        file_name = 'env.json'
        env_dict = {}
        if os.path.exists(file_name):
            with open(file_name) as json_content:
                env_dict = json.load(json_content)
        return env_dict

    def is_local_env(self) -> bool:
        return self.get_env()["env"] == "local"
