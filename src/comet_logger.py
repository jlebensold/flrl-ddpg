from comet_ml import Experiment
from .settings import COMET_API_KEY, COMET_WORKSPACE

class CometLogger:
    def __init__(self, project):
        self.project = project
        self.current_experiment = Experiment(
                api_key=COMET_API_KEY,
                workspace=COMET_WORKSPACE,
                project_name=self.project)

    def experiment(self):
        return self.current_experiment
