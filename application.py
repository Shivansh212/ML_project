from flask import Flask, request, render_template,redirect,url_for
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.Predict_pipeline import CustomData, predictpipeline

application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

## Route for prediction page
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # This will show the form page when the user navigates to /predict
        return render_template('home.html')
    else:
        # This block runs when the user submits the form
        data = CustomData(
            gender=request.form.get('gender'),
            # FIX 1: Changed 'race_ethnicity' to 'ethnicity' to match home.html
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            # FIX 2: Corrected the swapped keys for reading_score and writing_score
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
            
        )
        
        

        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        print("Before Prediction")
        predict_pipeline = predictpipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")

    
        
        return redirect(url_for("result", result=round(results[0], 2)))
    
@app.route('/result/<float:result>', methods=['GET', 'POST'])
def result(result):
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    # Note: For deployment on many platforms, you might remove the host and port
    # For local testing, this is fine.
    app.run(host='0.0.0.0', port=5001,debug=True)