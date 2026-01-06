from typing import Dict

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from common_utils import evaluate_predictions


def create_automl_voter(random_state: int = 42) -> VotingClassifier:
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=random_state)
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    svc = SVC(probability=True, random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=7)
    voter = VotingClassifier(
        estimators=[
            ("rf", rf), ("gb", gb), ("lr", lr), ("svc", svc), ("knn", knn)
        ],
        voting='soft',
        n_jobs=-1
    )
    return voter


def train_and_eval_automl(X_train, y_train, X_test, y_test) -> Dict[str, float]:
    model = create_automl_voter()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_predictions(y_test, y_pred)























