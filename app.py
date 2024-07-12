from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('liver.pkl', 'rb'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    age = int(request.form['age'])
    gender = 0 if request.form['gender'] == 'Male' else 1
    total_bilirubin = float(request.form['total_bilirubin'])
    direct_bilirubin = float(request.form['direct_bilirubin'])
    alkaline_phosphotase = int(request.form['alkaline_phosphotase'])
    alanine_aminotransferase = int(request.form['alanine_aminotransferase'])
    aspartate_aminotransferase = int(request.form['aspartate_aminotransferase'])
    total_protiens = float(request.form['total_protiens'])
    albumin = float(request.form['albumin'])
    albumin_and_globulin_ratio = float(request.form['albumin_and_globulin_ratio'])
    
    # Create a numpy array from the form data
    input_features = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                alanine_aminotransferase, aspartate_aminotransferase, total_protiens,
                                albumin, albumin_and_globulin_ratio]])
    
    # Make a prediction using the model
    prediction = model.predict(input_features)
    
    # Convert prediction to a human-readable format
    prediction_text = 'Liver Disease Detected' if prediction[0] == 1 else 'No Liver Disease Detected'
    
    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
