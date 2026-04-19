#  Tehran House Price Prediction Model  
### A Machine Learning and Deep Learning Pipeline for Real‑Estate Price Forecasting in Tehran

---

##  Overview
This project aims to **predict housing prices in Tehran** using a structured dataset containing property features such as area, age, location, and amenities.

A fully customized **PyTorch neural network (MLP)** was developed, trained using **Mini‑Batch Gradient Descent**, and evaluated using standard regression metrics such as **R²**, **MAE**, and **RMSE**.

The notebook includes a complete end‑to‑end pipeline:

- Data exploration  
- Preprocessing  
- Outlier handling  
- Feature scaling  
- Train / Validation / Test splitting  
- Model design and implementation  
- Training & optimization  
- Final evaluation  
- Visualization of results  

---

##  Notebook Structure

### 1. Importing Dependencies
The project uses:

- **NumPy, Pandas** for data manipulation  
- **Matplotlib, Seaborn** for visualization  
- **Scikit‑learn** for preprocessing and evaluation metrics  
- **PyTorch** for model building and training  

---

### 2. Data Loading & Initial Exploration
The dataset is loaded from a CSV file and inspected using:

- `df.head()`  
- `df.info()`  
- `df.describe()`  
- Histograms & boxplots to detect skewness and outliers  

---

## 🔧 3. Data Preprocessing

### Handling Missing Values
Rows containing missing values are removed:
```python
df = df.dropna()

### Handling Outliers
Price outliers are analyzed and managed using **IQR-based filtering** or **value capping** to prevent extreme values from dominating the training process.

### Feature Scaling
All numerical features are standardized using:

python
StandardScaler()

---

## 🔀 4. Dataset Splitting
The dataset is divided into three subsets:

- **Training set**
- **Validation set**
- **Test set** (used only for final evaluation)

Example split ratio:

text
Train:       70%
Validation:  15%
Test:        15%

---

## 🧠 5. Model Architecture (ManualRegression – PyTorch)

A custom neural network with two hidden layers is implemented:

python
class ManualRegression(nn.Module):
def __init__(self):
super().__init__()
self.fc1 = nn.Linear(18, 12)
self.fc2 = nn.Linear(12, 8)
self.fc3 = nn.Linear(8, 1)
self.relu = nn.ReLU()
self.dropout = nn.Dropout(0.005)

def forward(self, x):
x = self.relu(self.fc1(x))
x = self.dropout(x)
x = self.relu(self.fc2(x))
x = self.dropout(x)
x = self.fc3(x)
return x

### Architecture Summary
- **Input layer:** 18 features  
- **Hidden layer 1:** 12 neurons  
- **Hidden layer 2:** 8 neurons  
- **Output layer:** 1 neuron (price prediction)  
- **Activation:** ReLU  
- **Regularization:** Dropout (0.005)  
- **Loss function:** MSELoss  
- **Optimizer:** Mini‑Batch Gradient Descent (Adam / SGD)  

---

## 🏋️‍♂️ 6. Model Training (Mini‑Batch Gradient Descent)

The model is trained using:

- Forward pass  
- Backward pass (backpropagation)  
- Optimizer step  
- Loss tracking for **training** and **validation** sets  

Training is performed over multiple epochs while monitoring loss curves to ensure convergence and stability.

---

## 📊 7. Evaluation Metrics
After training, the model is evaluated **only on the test set** using:

- **R² Score** – Measures goodness of fit  
- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Squared Error)**  

Evaluation is done on **unscaled prices** for interpretability.

Example:

python
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

---

## 🚀 8. Final Training (Train + Validation)
to be continued


This approach:

- Maximizes learning capacity  
- Prevents data leakage  
- Keeps the test set completely unseen  

Final performance is then reported exclusively on the test set.

---

## 📉 9. Visualization
The notebook includes visualizations such as:

- Training & validation loss curves  
- Scatter plot of actual vs predicted prices  
- Error distribution plots  
- Histogram of predicted prices  

These plots help analyze model accuracy, bias, and error patterns.

---

## 🧾 Results Summary
The final model achieves approximately:

- **R² ≈ 0.80**  
- **Low MAE and RMSE**  
- Stable training without noticeable overfitting  
- Reasonably accurate price predictions  

Despite its lightweight architecture, the model performs well due to:

- Proper preprocessing  
- Log transformation of the target  
- Regularization  
- Mini‑Batch optimization  

---

## 🔧 Future Improvements
Potential enhancements include:

- Deeper neural network architectures  
- Increased dropout (0.1 – 0.2)  
- Hyperparameter tuning  
- Feature engineering (e.g., price per square meter)  
- Comparison with tree‑based models such as **XGBoost** or **LightGBM**

---

## ✅ Conclusion
This notebook provides a **complete, reliable, and well‑structured pipeline** for predicting housing prices in Tehran using a custom neural network.

It demonstrates strong engineering practices, clean preprocessing, stable training behavior, and solid evaluation results.

