from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from car_data_prep import prepare_data
import numpy as np
import os

app = Flask(__name__)

# Load the original dataset to extract unique values for dropdowns
df = pd.read_csv('C:\\Users\\guyul\\ProjectP3\\dataset.csv')

# Replace NaN values with 'לא מוגדר' (not defined)
df = df.replace({np.nan: 'לא מוגדר'})


# Extract unique values for dropdowns
unique_manufacturers = sorted(df['manufactor'].unique().tolist())
unique_models = sorted(df['model'].unique().tolist())
unique_gears = sorted(df['Gear'].unique().tolist())
unique_engine_types = sorted(df['Engine_type'].unique().tolist())
unique_prev_ownerships = sorted(df['Prev_ownership'].unique().tolist())
unique_curr_ownerships = sorted(df['Curr_ownership'].unique().tolist())
unique_areas = sorted(df['Area'].unique().tolist())
unique_cities = sorted(df['City'].unique().tolist())
unique_colors = sorted(df['Color'].unique().tolist())

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the mappings
with open('mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

# Reverse mappings
reverse_mappings = {key: {v: k for k, v in value.items()} for key, value in mappings.items()}

@app.route('/')
def index():
    # Render the index.html template with unique values for dropdowns
    return render_template('index.html', manufacturers=unique_manufacturers, models=unique_models, gears=unique_gears, engine_types=unique_engine_types, prev_ownerships=unique_prev_ownerships, curr_ownerships=unique_curr_ownerships, areas=unique_areas, cities=unique_cities, colors=unique_colors)

# Route to get models by manufacturer
@app.route('/get_models/<manufacturer>', methods=['GET'])
def get_models(manufacturer):
    # Filter models based on selected manufacturer
    models = df[df['manufactor'] == manufacturer]['model'].unique().tolist()
    return jsonify(models)


# Route to get cities by area
@app.route('/get_cities/<area>', methods=['GET'])
def get_cities(area):
    # Filter cities based on selected area
    cities = df[df['Area'] == area]['City'].unique().tolist()
    return jsonify(cities)


# Route to evaluate car price
@app.route('/evaluate_price', methods=['POST'])
def evaluate_price():
    # Get data from form submission
    data = request.form
    year = int(data['year'])
    manufacturer = mappings['manufacturer'][data['manufacturer']]
    model_car = mappings['model'][data['model']]
    gear = mappings['gear'][data['gear']]
    engine_type = mappings['engine_type'][data['engine_type']]
    prev_ownership = data['prev_ownership']
    curr_ownership = data['curr_ownership']
    color = mappings['color'][data['color']]
    hand = int(data['hand'])
    capacity_engine = int(data['capacity_engine'])
    km = int(data['km'])
    area = mappings['area'][data['area']]
    city = mappings['city'][data['city']]

    # Prepare the input for the model
    input_data = [[year, manufacturer, model_car, gear, engine_type, prev_ownership, curr_ownership, color, hand, capacity_engine, km, area, city]]
    input_df = pd.DataFrame(input_data, columns=['Year', 'manufacturer', 'model', 'gear', 'engine_type', 'Prev_ownership', 'Curr_ownership', 'Color', 'Hand', 'capacity_Engine', 'Km', 'area', 'city'])

    # One-hot encode the input data in the same way as the training data
    categorical_cols = ['manufacturer', 'model', 'Prev_ownership', 'Curr_ownership', 'Color', 'gear', 'engine_type', 'area', 'city']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Ensure all columns are present in the input data
    for col in scaler.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[scaler.feature_names_in_]
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(input_scaled)

    # Return the predicted price as a JSON response
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
