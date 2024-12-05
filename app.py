from flask import Flask, request, render_template, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_file_path = 'decision_tree_model.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        family_history = int(request.form['family_history'])
        smoking = int(request.form['smoking'])
        bmi = float(request.form['bmi'])
        exercise_hours = float(request.form['exercise_hours'])
        diabetes = int(request.form['diabetes'])
        obesity = int(request.form['obesity'])
        diet = int(request.form['diet'])
        systolic = int(request.form['systolic'])
        diastolic = int(request.form['diastolic'])

        # Prepare data for prediction
        new_data = np.array([[age, sex, family_history, smoking, bmi, exercise_hours, diabetes, obesity, diet, systolic, diastolic]])

        # Make prediction
        prediction = model.predict(new_data)
        predicted_category = 'Low Chance' if prediction[0] == 0 else 'High Chance'

        return render_template('result.html', prediction=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)