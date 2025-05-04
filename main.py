import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('diabetes.csv')

print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.Outcome.value_counts())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, random_state=10
)

dt_scores = cross_val_score(DecisionTreeClassifier(), X_scaled, y, cv=5)
print("Dokładność drzewa decyzyjnego (CV):", dt_scores.mean())

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)

bag_model.fit(X_train, y_train)

print("Dokładność OOB:", bag_model.oob_score_)

test_score = bag_model.score(X_test, y_test)
print("Dokładność na zbiorze testowym:", test_score)

bag_cv_scores = cross_val_score(bag_model, X_scaled, y, cv=5)
print("Bagging CV (średnia):", bag_cv_scores.mean())

rf_scores = cross_val_score(RandomForestClassifier(), X_scaled, y, cv=5)
print("Random Forest CV (średnia):", rf_scores.mean())

y_pred = bag_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność na zbiorze testowym (re-check): {test_accuracy:.4f}")
