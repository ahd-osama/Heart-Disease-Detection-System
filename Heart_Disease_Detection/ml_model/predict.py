import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

test = pd.read_csv("../data/test_data.csv")
actual = pd.read_csv("../data/test_labels.csv")

model = joblib.load("../data/model.pkl")

pred = model.predict(test)

pd.DataFrame(pred, columns=["Predicted"]).to_csv("/content/drive/My Drive/Heart_Disease_Detection/data/predictions.csv", index=False)

accuracy = accuracy_score(actual, pred)
precision = precision_score(actual, pred)
recall = recall_score(actual, pred)
f1 = f1_score(actual, pred)

print("Predictions saved successfully!")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

plt.figure(figsize=(60, 15),dpi=300)
plot_tree(model, feature_names=test.columns, class_names=["No Heart Disease", "Heart Disease"], filled=True, rounded=True ,
          fontsize=8)
plt.title("Decision Tree Visualization")
plt.savefig("/content/drive/My Drive/Heart_Disease_Detection/data/decision_tree.png", bbox_inches='tight')
plt.show()
