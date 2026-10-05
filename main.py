import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline de classification du churn")

    parser.add_argument(
        "--data", default="Churn_Modelling.csv", help="Chemin vers le fichier CSV"
    )

    parser.add_argument(
        "--model",
        default="classifier.joblib",
        help="Nom du fichier de sauvegarde du modèle",
    )

    args = parser.parse_args()

    print("=== Préparation des données ===")
    x_train, x_test, y_train, y_test = prepare_data(args.data)

    print("\n=== Entraînement du modèle ===")
    model = train_model(x_train, y_train)

    print("\n=== Évaluation du modèle ===")
    accuracy, matrix = evaluate_model(model, x_test, y_test)

    print("\nMatrice de confusion :")
    print(matrix)

    print("\n=== Sauvegarde du modèle ===")
    save_model(model, args.model)

    print("\n=== Pipeline terminé ===")


if __name__ == "__main__":
    main()
