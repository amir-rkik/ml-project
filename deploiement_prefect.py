"""Déploiements Prefect avec planification quotidienne."""
from prefect import serve

from pipeline_prefect import (
    all_flow,
    train_flow,
    evaluate_flow,
    code_flow,
    install_flow,
)

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",  # tous les jours à 02:00
            tags=["mlops", "full-pipeline"],
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train", tags=["mlops", "training"]
        ),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["mlops", "evaluation"]
        ),
        code_flow.to_deployment(
            name="ml-pipeline-code", tags=["mlops", "code"]
        ),
        install_flow.to_deployment(
            name="ml-pipeline-install", tags=["mlops", "setup"]
        ),
    )
