from openenv.client import EnvClient
from models import Observation, Action

class BearRaelClient(EnvClient):
    """
    Standard OpenEnv client for the BEAR-RAEL environment.
    """
    def __init__(self, base_url: str):
        super().__init__(base_url=base_url)

    def reset(self, seed: int = 42, task: str = "easy") -> Observation:
        resp = self._post("/reset", {"seed": seed, "task": task})
        return Observation(**resp)

    def step(self, action_type: str, parameters: dict = None) -> dict:
        payload = {"action_type": action_type, "parameters": parameters or {}}
        return self._post("/step", payload)

    def state(self) -> dict:
        return self._get("/state")
