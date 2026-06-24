# 📡 ChurnGuard — Telecom Customer Churn Prediction

A machine learning web app that predicts whether a telecom customer is likely to churn, built with **Streamlit** and **Scikit-Learn**.

---

##  Live Demo

> Run locally — see [Getting Started](#getting-started) below.

---

## 📸 Features

- **Instant churn risk prediction** with confidence probability
- **Risk Meter** — visual gauge from Low to High risk
- **Actionable alerts** — retention strategies for high-risk customers, upsell prompts for low-risk ones
- **Dark SaaS UI** with Inter & JetBrains Mono fonts
- **One-hot encoded pipeline** supporting 50 features across demographics, services, and billing

---

## 🗂 Project Structure

```
telecom-churn-prediction/
│
├── app.py                          # Streamlit application
├── model.sav                       # Trained ML model (pickle)
├── model.ipynb                     # Model training notebook
├── TeleCom_Customer_EDA.ipynb      # Exploratory Data Analysis notebook
├── tel_churn.csv                   # Preprocessed (one-hot encoded) dataset
├── Tele_Customer_Dataset-Copy1.csv # Raw telecom customer dataset
├── requirements.txt                # Python dependencies
└── README.md
```

---

##  Model Details

| Property        | Value                          |
|-----------------|--------------------------------|
| Task            | Binary Classification          |
| Framework       | Scikit-Learn                   |
| Features        | 50 (one-hot encoded)           |
| Target          | Churn (1) / No Churn (0)       |
| Input domains   | Demographics, Services, Billing|
| Output          | `predict_proba()` score        |

**Feature categories:**
- Demographics: Gender, Senior Citizen, Partner, Dependents, Tenure Group
- Services: Phone, Internet, Streaming TV/Movies, Online Security/Backup, Tech Support
- Billing: Contract type, Payment method, Monthly & Total charges, Paperless billing

---

## 🛠 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<Gaurav215b>/telecom-churn-prediction.git
cd telecom-churn-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Dataset

The dataset is based on the **IBM Telco Customer Churn** dataset, containing ~7,000 customer records with:
- Customer demographics
- Subscribed services
- Account information (contract, billing, charges)
- Churn label (Yes/No)

---

## 👤 Developer

**Gaurav**  
ML Engineer · Data Scientist  
📧 gaurav200b@gmail.com  
🔗 [github.com/Gaurav215b](https://github.com/Gaurav215b)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
