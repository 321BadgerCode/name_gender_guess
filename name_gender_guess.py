import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('./names.csv')

# TODO: preprocess the data if needed (e.g., remove duplicates, handle missing values)

X = df['Name']
y = df['Gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_features, y_train)

y_pred = model.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

new_names = ['John', 'Rick', 'Jessica', 'Ava']
new_names_features = vectorizer.transform(new_names)
predicted_genders = model.predict(new_names_features)
print(f'Predicted genders for {new_names}: {predicted_genders}')
