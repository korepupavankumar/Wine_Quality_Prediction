from data_loading import load_data
from preprocessing import create_target
from preprocessing import split_data
from preprocessing import scale_data

from models import logistic_model
from models import knn_model
from models import decision_tree_model

from evaluation import evaluate
from feature_importance import plot_importance


# Load data
df = load_data()

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())


# Create target
df = create_target(df)


# Split data
X_train, X_test, y_train, y_test = split_data(df)


# Scale data
X_train_scaled, X_test_scaled = scale_data(
    X_train,
    X_test
)


# Logistic Regression
lr = logistic_model()

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

print("\nLogistic Regression")
evaluate(y_test, lr_pred)


# KNN
knn = knn_model()

knn.fit(X_train_scaled, y_train)

knn_pred = knn.predict(X_test_scaled)

print("\nKNN")
evaluate(y_test, knn_pred)


# Decision Tree
dt = decision_tree_model()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree")
evaluate(y_test, dt_pred)


# Feature Importance
plot_importance(dt, X_train)

# User Input

while True:

    print("\nEnter Wine Details:")

    fixed_acidity = float(input("Fixed Acidity: "))
    volatile_acidity = float(input("Volatile Acidity: "))
    citric_acid = float(input("Citric Acid: "))
    residual_sugar = float(input("Residual Sugar: "))
    chlorides = float(input("Chlorides: "))
    free_sulfur_dioxide = float(input("Free Sulfur Dioxide: "))
    total_sulfur_dioxide = float(input("Total Sulfur Dioxide: "))
    density = float(input("Density: "))
    pH = float(input("pH: "))
    sulphates = float(input("Sulphates: "))
    alcohol = float(input("Alcohol: "))

    user_data = [[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]]

    prediction = dt.predict(user_data)

    if prediction[0] == 1:
        print("\nPrediction: GOOD WINE")
    else:
        print("\nPrediction: BAD WINE")

    choice = input(
        "\nDo you want to predict another wine? (yes/no): "
    ).lower()

    if choice != "yes":
        print("\nThank You!")
        break