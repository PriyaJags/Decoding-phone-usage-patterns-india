# 📱 Decoding Phone Usage Patterns in India

## Project Overview
An interactive data science project exploring smartphone usage patterns in India. This project uses exploratory data analysis (EDA), machine learning classification, and KMeans clustering to investigate usage behaviour, digital activities, and spending patterns.

## Objectives
- Explore smartphone usage trends through visualizations.
- Build a classification model to predict users' primary phone usage.
- Apply KMeans clustering to identify groups of users with similar usage and spending patterns.
- Present findings through an interactive Streamlit application.

## Technologies Used
- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Application:** Streamlit
- **Development:** Jupyter Notebook, VS Code

## Project Components
- `Decoding_phone_usage_patterns.ipynb` — Exploratory data analysis and visualizations.
- `Decoding_phone_usage_ML.ipynb` — Classification modelling and evaluation.
- `Decoding_phone_usage_clustering.ipynb` — Clustering analysis and user segmentation.
- `phone_usage_st.py` — Interactive Streamlit application.

## Key Findings
- The dataset contains 17,686 smartphone user records across five primary usage categories.
- The classification model achieved approximately 21% test accuracy, indicating limited predictive performance with the current features and modelling approach.
- KMeans clustering identified two broad user segments, with differences in average e-commerce spending.
- The clusters show substantial overlap, so the segments should be interpreted as exploratory rather than definitive user groups.

## How to Run the Application
1. Clone or download this repository.
2. Install the required Python libraries.
3. Ensure the required dataset and saved model files are available in the project directory.
4. Run the application:

   ```bash
   streamlit run phone_usage_st.py
   ```

## Limitations
- Classification performance is low and requires further improvement.
- Clustering results show limited separation between user groups.
- Findings describe patterns in the available dataset and should not be treated as proof of causal relationships.

## Future Improvements
- Explore additional classification models and feature engineering.
- Improve model evaluation and interpretation.
- Enhance the Streamlit dashboard with additional interactive insights.

---
**Note:** The large `random_forest_primary_use.pkl` model file is not included in this repository. The prediction feature requires the model to be supplied or regenerated locally.
