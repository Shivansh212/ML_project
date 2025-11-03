# End-to-End Student Performance Prediction

This is a complete end-to-end Machine Learning project that predicts a student's **Math Score** based on various demographic and academic factors.

The project includes data ingestion, data transformation, model training (with hyperparameter tuning), and a prediction pipeline, all wrapped in a web application and deployed on Render.

**🚀 Live Demo:** **[https://ml-project-5-w2yn.onrender.com](https://ml-project-5-w2yn.onrender.com)**

---

## 📈 Project Pipeline

This project follows a standard ML pipeline:


1.  **Data Ingestion:** Loads the `stud.csv` dataset, splits it into training and testing sets, and saves them in the `artifact/` directory.
2.  **Data Transformation:** A preprocessing pipeline (`ColumnTransformer`) is built to handle numerical and categorical features:
    * **Numerical Features:** `writing_score`, `reading_score`
        * Imputed with `median` values.
        * Scaled using `StandardScaler`.
    * **Categorical Features:** `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`
        * Imputed with the `most_frequent` value.
        * Encoded using `OneHotEncoder`.
        * Scaled using `StandardScaler` (with `with_mean=False`).
3.  **Model Training:**
    * Trains 7 different regression models to find the best performer:
        * Random Forest
        * Decision Tree
        * Gradient Boosting
        * Linear Regression
        * XGBRegressor
        * CatBoost Regressor
        * AdaBoost Regressor
    * Uses `GridSearchCV` to find the best hyperparameters for each model.
    * Evaluates models based on their **R² score**.
    * The best-performing model is saved as `model.pkl`.
4.  **Prediction Pipeline:**
    * A `Predict_pipeline.py` script loads the `preprocessor.pkl` and `model.pkl` artifacts.
    * A `CustomData` class maps input from the web form to the format required by the model.
5.  **Web Application:**
    * A Flask (or FastAPI) app serves an HTML page to get user inputs.
    * It uses the prediction pipeline to return the predicted math score.

---

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy:** For data manipulation.
* **Scikit-learn:** For the entire ML pipeline (splitting, preprocessing, modeling).
* **XGBoost, CatBoost, AdaBoost:** For advanced gradient-boosting models.
* **dill:** For serializing (saving) the preprocessing and model objects.
* **Flask / FastAPI:** For serving the web application.
* **Render:** For cloud deployment.
