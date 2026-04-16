# 🚦 Smart Traffic Control System (ML-Based)

This project implements a machine learning-based smart traffic control system using AWS services.

## 🔧 Features

* Data preprocessing from raw traffic data (AWS S3)
* Machine Learning models:

  * Linear Regression
  * Random Forest
* Real-time inference for traffic light optimization
* Integration with AWS IoT for live control

## 🧠 Architecture

1. Raw data stored in S3
2. Preprocessing using AWS Lambda
3. Model training (LR & RF)
4. Real-time inference
5. Publish result to IoT device

## 📁 Project Structure

* `preprocessing/` → data cleaning
* `training/` → ML model training
* `inference/` → real-time prediction

## ⚙️ Tech Stack

* Python
* AWS Lambda
* AWS S3
* AWS IoT
* Scikit-learn

## 🚀 Author

Nabil Raditya