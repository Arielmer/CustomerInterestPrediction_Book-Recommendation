# Charles Book Club - Customer Purchase Prediction

## Project Overview
This project aims to predict customer interest in purchasing the book *The Magician of Florence* based on their past purchase behavior. Charles Book Club, a subscription-based book service, faced declining profit margins despite increasing customer engagement. To address this, we developed a machine learning model to identify customers most likely to make a purchase.

## Dataset Description
The dataset consists of customer transaction data from a promotional campaign. It includes demographic information, purchase history, and behavioral attributes. 

### Key Variables:
- `Seq#`: Sequence number
- `ID#`: Customer identification number
- `Gender`: Gender (0 = Male, 1 = Female)
- `M`: Monetary - Total book purchases amount
- `R`: Recency - Months since the last purchase
- `F`: Frequency - Purchase frequency
- `FirstPurch`: Months since the first purchase
- `ChildBks` - `Florence`: Various book category purchase counts
- `Related Purchase`: Number of related book purchases
- `Florence`: Target variable (1 = Purchased, 0 = Did not purchase)

## Methodology
Several machine learning classification models were applied to predict customer interest, including:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Decision Tree**
- **Random Forest** (Best-performing model)

### Techniques Used:
- **Data Cleaning**: Removing missing values and handling categorical variables.
- **Feature Selection**: Using RFM (Recency, Frequency, Monetary) analysis to determine influential predictors.
- **Class Imbalance Handling**: Applying SMOTE (Synthetic Minority Over-sampling Technique) to improve minority class prediction.
- **Model Evaluation**: Comparing accuracy, precision, recall, and F1-score to select the best model.

## Results
The **Random Forest model with SMOTE** provided the best balance of precision and recall for predicting customer purchases. The model successfully identified high-potential customers for targeted marketing efforts.

### Key Visualizations:
#### Feature Importance (Random Forest)
The following bar chart highlights the top 10 most important features in the Random Forest model, showing which variables had the most impact on the purchase prediction.

![Top 10 Important Features - Random Forest](feature_importance.png)

#### Confusion Matrix (Random Forest)
The confusion matrix illustrates the model's performance in classifying purchasing and non-purchasing customers. 

![Confusion Matrix - Random Forest](confusion_matrix.png)

## How to Use the Jupyter Notebook
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook charlesbook_annotated.ipynb
   ```
3. Follow the structured code to:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Train and evaluate classification models
   - Compare model performance
   - Generate final business insights

## Conclusion
This project demonstrates how machine learning can optimize marketing strategies by identifying customers most likely to purchase new book offerings. The results enable Charles Book Club to implement targeted marketing campaigns and increase profitability.

