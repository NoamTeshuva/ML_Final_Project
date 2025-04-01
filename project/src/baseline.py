import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ✅ Load test data
test_df = pd.read_csv("C:/Users/HP/PycharmProjects/ML_final_project/data/selected_200/testing_data.csv")

# ✅ Sort by date to ensure proper temporal order
test_df["Date"] = pd.to_datetime(test_df["Date"])
test_df = test_df.sort_values("Date")

# ✅ Create baseline prediction
test_df["Baseline_Pred"] = test_df["Target"].shift(1)
test_df.dropna(inplace=True)
test_df["Baseline_Pred"] = test_df["Baseline_Pred"].astype(int)

# ✅ Evaluate baseline
baseline_acc = accuracy_score(test_df["Target"], test_df["Baseline_Pred"])
conf_matrix = confusion_matrix(test_df["Target"], test_df["Baseline_Pred"])

print(f"📉 Baseline Accuracy (Yesterday's trend): {baseline_acc:.2%}")
print("✅ Confusion Matrix:")
print(conf_matrix)

# ✅ Create visuals output folder
import os
visual_dir = "C:/Users/HP/PycharmProjects/ML_final_project/visuals"
os.makedirs(visual_dir, exist_ok=True)

# ✅ Plot accuracy
plt.figure(figsize=(5, 4))
sns.barplot(x=["Baseline"], y=[baseline_acc], palette="Blues")
plt.ylim(0.4, 0.7)
plt.ylabel("Accuracy")
plt.title("Baseline Accuracy (Yesterday's Trend)")
plt.tight_layout()
plt.savefig(os.path.join(visual_dir, "baseline_accuracy_bar.png"))
plt.show()

# ✅ Plot confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Down", "Up"])
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title("Baseline Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(visual_dir, "baseline_confusion_matrix.png"))
plt.show()

print("✅ Visualizations saved to /visuals/")
