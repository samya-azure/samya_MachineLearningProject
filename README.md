
# 📦 Demand Forecasting ML API

This project provides a machine learning API using FastAPI that predicts **whether a customer will order a product in the next 7 days** based on recent order patterns. It uses a Random Forest Classifier trained on customer-item transaction data.

---

## 💼 Use Case

Forecast demand for specific product(s) per customer in upcoming days/weeks. This is useful for:

- Inventory planning 📦
- Personalized marketing 🎯
- Supply chain optimization 🚚

---

## 🧠 Features Used for Prediction

- `days_since_last_order`
- `total_orders_last_30_days`
- `avg_order_interval`
- `recently_viewed`

These features are derived from historical interactions.

---

## 📁 Project Structure

```
demand_forecasting_app/
├── app/                  # FastAPI routes and request handling
│   ├── main.py
│   └── schemas.py
├── core/                 # ML logic: training and prediction
│   ├── train_model.py
│   └── predict.py
├── data/                 # Input Excel dataset
│   └── customer_item_orders_sample.xlsx
├── models/               # Trained ML model saved here
│   └── model.pkl
├── tests/                # Unit and integration tests
│   └── test_api.py
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

---

## 🚀 How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (if needed)

```bash
python core/train_model.py
```

### 3. Run the FastAPI server

```bash
uvicorn app.main:app --reload
```

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI.

---

## 🧪 Sample API Usage

### Request:
```json
POST /predict

{
  "customer_codes": ["C001", "C002"],
  "item_codes": ["P001", "P004"]
}
```

### Response:
```json
{
  "predictions": [
    {
      "customer_code": "C001",
      "item_code": "P001",
      "prediction": "Will Order"
    },
    {
      "customer_code": "C002",
      "item_code": "P004",
      "prediction": "Will Not Order"
    }
  ]
}
```

---

## 📊 Model Info

- Model: RandomForestClassifier
- Target: `ordered_in_next_7_days` (1 = Will Order, 0 = Will Not Order)
- Accuracy and metrics shown after training

---

## 👤 Author

> Built with ❤️ by Samya Basu

---

## 🛠️ Future Enhancements

- [ ] Add database support (e.g., MongoDB, PostgreSQL)
- [ ] Add authentication for API access
- [ ] Add product and customer embedding features
- [ ] Create dashboard for visualizing demand trends
