# End-to-End Diabetes Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Model](https://img.shields.io/badge/Model-XGBoost-orange)
![Deployment](https://img.shields.io/badge/App-Streamlit%2FFlask-green)

## 📖 Overview
This project implements a complete machine learning pipeline to predict the likelihood of diabetes in patients based on diagnostic measures. 

The solution covers the entire workflow from **Exploratory Data Analysis (EDA)** and data preprocessing to **Model Training** using XGBoost and **Deployment** via a user-friendly web interface. It is designed to demonstrate a production-ready approach to medical classification tasks.

## 📂 Project Structure
* **`End_to_End_Diabetes_Prediction.ipynb`**: The main notebook containing dataset loading, in-depth EDA, feature engineering, and initial model experimentation.
* **`Xgboost_train.py`**: A modular Python script dedicated to training the final XGBoost model and saving the artifacts for production.
* **`app.py`**: The web application file that loads the trained model and provides a frontend for users to input data and get predictions.
* **`requirements.txt`**: Lists all the Python dependencies required to run the project.

## 📊 Dataset
The dataset handling and loading mechanisms are integrated directly within the **`End_to_End_Diabetes_Prediction.ipynb`** notebook. 
* The data includes medical predictor variables such as **Glucose Level**, **Blood Pressure**, **Insulin**, **BMI**, and **Age**.
* Refer to the notebook to view the data sourcing and preprocessing steps.

## 🛠️ Technologies Used
* **Python**: Core programming language.
* **Pandas & NumPy**: Data manipulation and numerical operations.
* **XGBoost**: Gradient boosting framework used for high-performance classification.
* **Scikit-Learn**: Used for preprocessing, pipelines, and evaluation metrics.
* **Web Framework**: (Streamlit or Flask) used for `app.py`.

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/mdjahirulislam56/Diabetes-Risk-Prediction-ML-Project.git](https://github.com/mdjahirulislam56/Diabetes-Risk-Prediction-ML-Project.git)
cd Diabetes-Risk-Prediction-ML-Project

```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt

```

### 3. Train the Model

To retrain the model and generate the necessary artifacts:

```bash
python Xgboost_train.py

```

### 4. Run the Web App
Launch the Gradio interface:

```bash
python app.py

```
The app will launch in your browser (usually at http://127.0.0.1:7860).

## 📈 Results

The project utilizes **XGBoost** for its efficiency and accuracy on structured data. Detailed performance metrics, including Accuracy, Precision, Recall, and the Confusion Matrix, are available in the analysis notebook.

## 📝 Author

**Md Jahirul Islam**

---

*This project serves as a demonstration of an end-to-end machine learning workflow, from raw data to a deployed application.*

```

```
