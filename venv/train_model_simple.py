import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading data...")
df = pd.read_csv('data/Salary Data.csv')
df = df.dropna()
print(f"Data shape: {df.shape}")

print("Encoding categoricals...")
le = LabelEncoder()
for col in ['Gender', 'Education Level', 'Job Title']:
    df[col] = le.fit_transform(df[col].astype(str))

X = df[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title']]
y = df['Salary']

print("Splitting data...")
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Saving model...")
joblib.dump(model, 'model/salary_predictor__corrected.pkl')
print("Model trained and saved successfully!")

# Test prediction
test_data = {'Age': 32, 'Gender': 'Male', 'Education Level': "Master's", 'Years of Experience': 5, 'Job Title': 'Software Engineer'}
df_test = pd.DataFrame([test_data])
pred = model.predict(df_test)[0]
print(f"Test prediction: ${pred:,.2f}")

