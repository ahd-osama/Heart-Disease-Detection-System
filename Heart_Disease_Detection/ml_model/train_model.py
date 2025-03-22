import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib

data = pd.read_csv("../data/cleaned_data.csv")

actualdata = data.drop(columns=['target'])
labels = data['target']

actualdata_train, actualdata_test, labels_train, labels_test = train_test_split(actualdata, labels, test_size=0.2, random_state=42, stratify=labels)

actualdata_test.to_csv("../data/test_data.csv", index=False)
labels_test.to_csv("../data/test_labels.csv", index=False)

clf = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(actualdata_train, labels_train)

best_model = grid_search.best_estimator_
best_model.fit(actualdata_train, labels_train)

joblib.dump(best_model, "../data/model.pkl")
print("Model saved successfully!")
