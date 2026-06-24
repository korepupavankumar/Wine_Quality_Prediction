import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_importance(model, X):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance":
        model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance)

    plt.figure(figsize=(8,5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance
    )

    plt.title("Feature Importance")

    plt.show()