import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(data_filename="Churn_Modelling (1).csv"):
    """
    Charge et prépare les données pour l'entraînement du modèle.
    """

    df = pd.read_csv(data_filename)

    # Encoder la colonne Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Supprimer les colonnes inutiles
    columns_to_drop = ["Surname", "Geography"]
    df = df.drop(columns=columns_to_drop)

    # Séparer les variables explicatives et la variable cible
    x = df.drop(columns=["Exited"])
    y = df["Exited"]

    # Supprimer les identifiants
    x = x.drop(columns=["RowNumber", "CustomerId"])

    # Diviser les données en entraînement et test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=1,
    )

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """
    Entraîne le modèle Random Forest.
    """

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    rf.fit(x_train, y_train)

    return rf


def evaluate_model(model, x_test, y_test):
    """
    Évalue les performances du modèle.
    """

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy score:", accuracy * 100)

    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, matrix


def save_model(model, filename="classifier.joblib"):
    """
    Sauvegarde le modèle entraîné.
    """

    joblib.dump(model, filename)

    print(f"Modèle sauvegardé dans : {filename}")


def load_model(filename="classifier.joblib"):
    """
    Charge un modèle sauvegardé.
    """

    model = joblib.load(filename)

    print(f"Modèle chargé depuis : {filename}")

    return model


def predict(model, features):
    """Prédit la classe pour un client.

    Args:
        model: modèle entraîné.
        features (dict): caractéristiques du client (9 colonnes).

    Returns:
        int: classe prédite (0 = reste, 1 = quitte la banque).
    """
    import pandas as pd

    columns = [
        "CreditScore",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    frame = pd.DataFrame([features], columns=columns)
    return int(model.predict(frame)[0])
