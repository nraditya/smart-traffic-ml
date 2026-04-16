# 🚦 Smart Traffic Control System (ML + AWS + IoT)

A machine learning-based smart traffic light control system that dynamically adjusts green light duration using real-time traffic data.

---

## 🧠 Key Features

* Automated data preprocessing from AWS S3
* Machine learning models:

  * Linear Regression
  * Random Forest
* Real-time inference using AWS Lambda
* IoT integration for traffic light control
* Hybrid decision system (real + predicted speed)

---

## 🏗️ System Architecture

1. Raw traffic data stored in AWS S3
2. Data cleaning using Lambda (preprocessing)
3. Model training (LR & RF)
4. Real-time inference
5. Publish decision to IoT device

---

## 📁 Project Structure

* `preprocessing/` → data cleaning pipeline
* `training/` → ML model training
* `inference/` → real-time prediction & IoT publishing

---

## ⚙️ Tech Stack

* Python
* AWS Lambda
* AWS S3
* AWS IoT Core
* Scikit-learn

---

## 📊 Machine Learning

* Features:

  * distance_m
  * duration_s
  * duration_in_traffic_s
* Target:

  * speed_kmh

---


## 👤 Author

Nabil Raditya
