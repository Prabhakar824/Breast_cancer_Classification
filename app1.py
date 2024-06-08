from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('br_cancer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Extract form data
            data = {
                'radius_mean': float(request.form['radius_mean']),
                'texture_mean': float(request.form['texture_mean']),
                'perimeter_mean': float(request.form['perimeter_mean']),
                'area_mean': float(request.form['area_mean']),
                'smoothness_mean': float(request.form['smoothness_mean']),
                'compactness_mean': float(request.form['compactness_mean']),
                'concavity_mean': float(request.form['concavity_mean']),
                'concave_points_mean': float(request.form['concave_points_mean']),
                'symmetry_mean': float(request.form['symmetry_mean']),
                'fractal_dimension_mean': float(request.form['fractal_dimension_mean']),
                'radius_se': float(request.form['radius_se']),
                'texture_se': float(request.form['texture_se']),
                'perimeter_se': float(request.form['perimeter_se']),
                'area_se': float(request.form['area_se']),
                'smoothness_se': float(request.form['smoothness_se']),
                'compactness_se': float(request.form['compactness_se']),
                'concavity_se': float(request.form['concavity_se']),
                'concave_points_se': float(request.form['concave_points_se']),
                'symmetry_se': float(request.form['symmetry_se']),
                'fractal_dimension_se': float(request.form['fractal_dimension_se']),
                'radius_worst': float(request.form['radius_worst']),
                'texture_worst': float(request.form['texture_worst']),
                'perimeter_worst': float(request.form['perimeter_worst']),
                'area_worst': float(request.form['area_worst']),
                'smoothness_worst': float(request.form['smoothness_worst']),
                'compactness_worst': float(request.form['compactness_worst']),
                'concavity_worst': float(request.form['concavity_worst']),
                'concave_points_worst': float(request.form['concave_points_worst']),
                'symmetry_worst': float(request.form['symmetry_worst']),
                'fractal_dimension_worst': float(request.form['fractal_dimension_worst']),
            }
            
            # Make prediction using the model
            features = np.array([list(data.values())])
            model_prediction = model.predict(features)

            result = "Malignant" if model_prediction[0] == 0 else "Benign"

            return render_template('result.html', prediction=result)
        except KeyError as e:
            return f"KeyError: {e} is missing from the form data", 400

    return render_template('home1.html')

@app.route('/predict', methods=['GET'])
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)

