import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, request, jsonify, render_template

# Sample dataset
def load_real_estate_data():
    data = pd.DataFrame({
        'size': [1500, 2000, 2500, 3000, 3500],
        'bedrooms': [3, 4, 4, 5, 5],
        'bathrooms': [2, 3, 3, 4, 4],
        'age': [10, 15, 20, 25, 30],
        'location_score': [8, 9, 7, 8, 9],
        'price': [300000, 400000, 500000, 600000, 700000]
    })
    return data

# Load dataset
data = load_real_estate_data()
X = data.drop('price', axis=1)
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
]

meta_model = LinearRegression()

# Stacking Regressor
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train model
stacking_regressor.fit(X_train, y_train)

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input features
        size = float(request.form['size'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        age = int(request.form['age'])
        location_score = float(request.form['location_score'])
        
        # Prepare data for prediction
        features = pd.DataFrame([[size, bedrooms, bathrooms, age, location_score]],
                                 columns=['size', 'bedrooms', 'bathrooms', 'age', 'location_score'])
        
        # Make prediction
        prediction = stacking_regressor.predict(features)[0]
        
        return render_template('result.html', prediction=round(prediction, 2))
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
