import os
import mlflow
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()

def configure_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("[WARN] MLFLOW_TRACKING_URI not set — logging to local ./mlruns instead")

@contextmanager
def mlflow_run(run_name: str, experiment_name: str = "legal-case-search"):
    configure_mlflow()
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        yield run