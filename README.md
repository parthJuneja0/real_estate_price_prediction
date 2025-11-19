# 🏠 Real Estate Price Prediction (Taiwan Housing Dataset)

This project builds a **machine learning model** to predict **house price per unit area** using the Taiwan real-estate dataset (features X1–X6).  
It includes:

- 📥 Automatic train/test split using **stratified sampling on house-price bins**
- ⚙️ Data preprocessing pipeline with **imputation + scaling**
- 🌲 A **Random Forest Regressor** trained on the stratified training set
- 📊 Evaluation using **Test RMSE** and **Cross-validation**
- 🎯 Real prediction on **unseen test samples**
- ✍️ Custom input prediction support
- 💾 Saving/loading model using `joblib`

---

## 📂 Project Structure

```
real_estate_price_prediction/
│
├── data/
│ └── data.csv # Original dataset (X1–X6, Y)
│
├── models/
│ └── Dragon.joblib # Saved trained model
│
├── main.py # Main training/evaluation/prediction pipeline
└── README.md
```

---

## 🔢 Dataset Information

Your dataset contains these columns:

| Column | Meaning |
|--------|---------|
| X1 | Transaction date |
| X2 | House age |
| X3 | Distance to nearest MRT station |
| X4 | Number of convenience stores |
| X5 | Latitude |
| X6 | Longitude |
| Y  | **House price per unit area** (target variable) |

---
