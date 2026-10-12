"""Pipeline MLOps orchestré avec Prefect."""

import argparse
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
# On ne cible que nos fichiers (sinon le scan du venv est très long)
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py", "app.py"]


def run_cmd(cmd):
    """Exécute une commande et lève une erreur si elle échoue."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Échec : {' '.join(cmd)}")


# ---------- TASKS : suivi du code ----------
@task(name="Installer les dépendances")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code (black)")
def format_code():
    run_cmd([sys.executable, "-m", "black", *FILES])


@task(name="Qualité du code (flake8)")
def check_quality():
    run_cmd([sys.executable, "-m", "flake8", "--max-line-length=88", *FILES])


@task(name="Sécurité du code (bandit)")
def check_security():
    run_cmd([sys.executable, "-m", "bandit", "-ll", "-q", *FILES])


@task(name="Tests unitaires (pytest)")
def run_tests():
    run_cmd([sys.executable, "-m", "pytest", "tests", "-v"])


# ---------- TASKS : data / modèle ----------
@task(name="Préparation des données")
def task_prepare():
    return prepare_data(DATA_PATH)


@task(name="Entraînement du modèle")
def task_train(x_train, y_train):
    return train_model(x_train, y_train)


@task(name="Sauvegarde du modèle")
def task_save(model):
    save_model(model, MODEL_PATH)


@task(name="Chargement du modèle")
def task_load():
    return load_model(MODEL_PATH)


@task(name="Évaluation du modèle")
def task_evaluate(model, x_test, y_test):
    accuracy, matrix = evaluate_model(model, x_test, y_test)
    print(f"Accuracy : {accuracy}")
    print(matrix)
    return accuracy


@task(name="Récupération du projet (git)")
def git_pull():
    run_cmd(["git", "pull", "origin", "main"])


@task(name="Lancement de l'API (FastAPI)")
def start_api():
    print("API sur http://127.0.0.1:8000/docs (Ctrl+C pour arrêter)")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "0.0.0.0",  # nosec B104
            "--port",
            "8000",
        ],
        check=False,
    )


@flow(name="api")
def api_flow():
    start_api()


# ---------- FLOWS ----------
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = task_prepare()
    model = task_train(x_train, y_train)
    task_save(model)


@flow(name="evaluate")
def evaluate_flow():
    _, x_test, _, y_test = task_prepare()
    model = task_load()
    task_evaluate(model, x_test, y_test)


@flow(name="all")
def all_flow():
    git_pull()
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()
    x_train, x_test, y_train, y_test = task_prepare()
    model = task_train(x_train, y_train)
    task_save(model)
    task_evaluate(model, x_test, y_test)


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
    "install": install_flow,
    "api": api_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
