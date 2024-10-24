import sys

from metaflow import FlowSpec, Parameter, card, current, project, step
from metaflow.cards import Image, Table


def confusion_matrix_fig(y_real, y_pred):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_real, y_pred)
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    return fig


@project(name="doppler")
class DemoFlow(FlowSpec):
    """
    A flow to show the team Metaflow features.

    """

    random_state = Parameter("seed", default=42)
    features_dim = Parameter("features-dim", default=4)

    @step
    def start(self):
        """
        This is the 'start' step. All flows must have a step named 'start' that
        is the first step in the flow.

        """
        print("DemoFlow is starting.")
        print("Errors are shown in a different section", file=sys.stderr)

        self.next(self.prepare_data)

    @card(type="blank")
    @step
    def prepare_data(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        dataset_bunch = datasets.load_iris(as_frame=True)
        X = dataset_bunch["data"]
        y = dataset_bunch["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=self.random_state
        )
        self.X_train = X_train
        current.card.append(
            Table(self.X_train.values.tolist(), headers=self.X_train.columns.tolist())
        )
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.feature_names = dataset_bunch["feature_names"]
        print("Dataset available features:", len(self.feature_names))
        self.next(self.feature_selection)

    @step
    def feature_selection(self):
        from itertools import combinations

        self.feature_combinations = [
            list(combination)
            for combination in combinations(self.feature_names, self.features_dim)
        ]
        print("Number of feature combinations to test:", len(self.feature_combinations))
        self.next(self.train, foreach="feature_combinations")

    @step
    def train(self):
        # The feature_combination currently being processed is a class property called
        # 'input'.
        self.feature_combination = self.input
        print("Features to train", self.feature_combination)

        self.next(self.train_rf, self.train_xgb)

    @card(type="blank")
    @step
    def train_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score

        self.model_name = "Random Forest"
        self.clf = RandomForestClassifier(random_state=self.random_state).fit(
            self.X_train, self.y_train
        )
        self.y_pred = self.clf.predict(self.X_test)
        current.card.append(
            Image.from_matplotlib(confusion_matrix_fig(self.y_test, self.y_pred))
        )
        self.score = float(f1_score(self.y_test, self.y_pred, average="macro"))
        self.next(self.join_train)

    @step
    def train_xgb(self):
        from sklearn.metrics import f1_score
        from xgboost import XGBClassifier

        self.model_name = "XGBoost"
        self.clf = XGBClassifier(random_state=self.random_state).fit(
            self.X_train, self.y_train
        )
        self.y_pred = self.clf.predict(self.X_test)
        self.score = round(
            float(f1_score(self.y_test, self.y_pred, average="macro")), 4
        )
        self.next(self.join_train)

    @step
    def join_train(self, inputs):
        self.scores = []
        for i in inputs:
            self.scores.append([i.model_name, i.feature_combination, i.score])
        self.next(self.join)

    @step
    def join(self, inputs):
        all_scores = []
        for i in inputs:
            all_scores.extend(i.scores)
        all_scores = sorted(all_scores, key=lambda x: x[2], reverse=True)
        self.all_scores = all_scores
        self.best_model = all_scores[0]
        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        print("DemoFlow is done!")


if __name__ == "__main__":
    DemoFlow()
