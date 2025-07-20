# Financial Fraud Detection System

## 🌟 Project Overview

This project develops a robust **Financial Fraud Detection System** leveraging machine learning techniques to identify and prevent fraudulent transactions. By analyzing a comprehensive dataset of financial transactions, the system aims to detect anomalies and patterns indicative of fraudulent activities, thereby safeguarding financial operations.

## ✨ Key Features & Technologies

* **Machine Learning Model:** Utilizes a **RandomForestClassifier**, a powerful ensemble learning method, for accurate classification of transactions as legitimate or fraudulent.
* **Data Preprocessing:** Implements `StandardScaler` for numerical feature scaling and `OneHotEncoder` for efficient handling of categorical variables, optimizing data for model training.
* **Feature Engineering:** Creates insightful features like `balanceDiffOrig` and `balanceDiffDest` to better capture behavioral patterns related to fraud.
* **Comprehensive Data Handling:** Processes a substantial financial dataset (e.g., millions of entries) including diverse transaction details such as `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, and `isFlaggedFraud`.
* **Performance Evaluation:** Employs standard machine learning metrics including `classification_report`, `confusion_matrix`, `roc_auc_score`, and `roc_curve` to thoroughly assess model performance and reliability.
* **Predictive Functionality:** Provides a dedicated function to predict fraud for new transaction inputs, offering both a binary prediction (FRAUD/NOT FRAUD) and an associated probability.
* **Libraries Used:**
    * `pandas`
    * `numpy`
    * `scikit-learn` (for models, preprocessing, and metrics)
    * `matplotlib`
    * `seaborn`

## ⚙️ How It Works

The system operates in several key phases:

1.  **Data Loading & Exploration:** The initial step involves loading the raw transaction data, performing exploratory data analysis (EDA) to understand data distributions, identify missing values, and gain insights into transaction types and their relationships with fraud.
2.  **Data Cleaning & Preprocessing:** Handles missing values (if any), encodes categorical features (e.g., 'type' of transaction), and scales numerical features to ensure they are on a comparable scale, which is vital for many machine learning algorithms.
3.  **Feature Engineering:** New features are derived from existing ones to provide the model with more discriminative information for fraud detection.
4.  **Model Training:** A RandomForestClassifier is trained on the preprocessed dataset. The model learns to identify complex patterns and relationships that distinguish fraudulent transactions from legitimate ones.
5.  **Model Evaluation:** The trained model is rigorously evaluated using a separate test set. Performance metrics provide a quantitative measure of the model's accuracy, precision, recall, and overall effectiveness.
6.  **Prediction:** The trained model can then be used to predict fraud on new, unseen transaction data, providing real-time or batch fraud detection capabilities.

## 📊 Dataset

The project utilizes a synthetic dataset containing financial transaction records, typically found as `AIML Dataset.csv`. This dataset includes various transaction attributes, crucial for building a robust fraud detection model.

## 🚀 Getting Started

To run this project locally, follow these steps:

### Prerequisites

* Python 3.x
* Jupyter Notebook (recommended for running the `.ipynb` file)
* Required Python packages (install via `pip`):
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```

### Installation

1.  Clone the repository:
    ```bash
    [[git clone [https://github.com/YOUR_GITHUB_USERNAME/Financial-Fraud-Detection-System.git](https://github.com/YOUR_GITHUB_USERNAME/Financial-Fraud-Detection-System.git)
    cd Financial-Fraud-Detection-System](https://github.com/Bishal-Nengminja/Financial_Fraud_Detection_System/blob/main/Financial_Fraud_Detection_System.ipynb)](https://github.com/Bishal-Nengminja/Financial_Fraud_Detection_System)
    ```
2.  Place your `AIML Dataset.csv` file in the root directory of the cloned repository.
3.  Open the Jupyter Notebook:
    ```bash
    [jupyter notebook Financial_Fraud_Detection_System.ipynb](https://colab.research.google.com/drive/1k2syqA0UGAn6Xgyrko24A3ZrDKcobgTO)
    ```
4.  Run all cells in the notebook to execute the entire fraud detection pipeline.

## 📈 Results and Performance

(You might want to add specific performance metrics here once you've run the notebook and have them handy, e.g., "The model achieved an AUC-ROC score of XX%..." or "A precision of XX% and recall of YY% for fraudulent transactions.")

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.
(Note: You might need to create a `LICENSE` file if you haven't already. MIT License is a common choice for open-source projects.)

## 📞 Contact

Bishal Nengmnja
bishalnengminja61@gmail.com
