Bike Sharing Demand Prediction

A Machine Learning Regression Project using Python, CatBoost, and Streamlit

This project predicts hourly bike rental demand using the Bike Sharing Dataset (hour.csv).
The system includes:
Data loading & preprocessing (utils.py)
Multiple machine learning models (CatBoost, RandomForest, Linear Regression, etc.)
Model evaluation (MAE, RMSE, R²)
A Streamlit web app for interactive predictions
Version control with meaningful commits across weekly milestones
MIT License included

Dataset Source: UCI Bike Sharing Dataset
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

⚙️ Installation
git clone https://github.com/Iminokhun/cw_ml.git
cd cw_ml

Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

📦 Install dependencies
pip install -r requirements.txt

Run the Streamlit App
streamlit run app/streamlit_app.py

