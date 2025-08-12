import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "Product Name": "Smartwatch",
    "Category": "Electronics",
    "Price": 100,
    "Quantity": 5,
    "Customer Age": 50,
    "Customer Gender": "Male",
    "Discount": 0.25,
    "Payment Method": "Debit Card"   
}

response = requests.post(url, json=data)
if response.status_code == 200:
    print("Predicción:", response.json())
else:
    print("Error:", response.status_code, response.text)

