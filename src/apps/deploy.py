import modal

from src.apps.evaluation.app import app as evaluation_app
from src.apps.inference_engine.app import app as inference_engine_app
from src.apps.rollouts.app import app as rollouts_app
from src.apps.trainer.app import app as trainer_app

app = (
    modal.App("diplomacy-grpo")
    .include(inference_engine_app)
    .include(rollouts_app)
    .include(trainer_app)
    .include(evaluation_app)
)
