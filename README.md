# Calories Burned Prediction Model 

## Project Overview  
This project focuses on predicting **calories burned** during physical activities using **advanced machine learning techniques**. The goal is to develop an **accurate** and **efficient** predictive model that provides insights into **energy expenditure** based on key physiological factors.  

The model **outperformed benchmark standards** with the following performance metrics:  

### Best Model Performance Metrics  
- **Mean Absolute Error (MAE)**: 1.003  
- **R-Squared Score (R²)**: 0.999  
- **Mean Squared Error (MSE)**: 2.035  
- **Root Mean Squared Error (RMSE)**: 1.426  

## Applications  
This predictive model has a wide range of real-world applications, including:  

- **Fitness & Health Tracking** – Personalized fitness recommendations based on calorie expenditure.  
- **Smart Wearables** – Integration with smartwatches and fitness bands for real-time calorie estimation.  
- **Sports Performance Analysis** – Helps athletes optimize training by understanding energy expenditure.  
- **Healthcare & Nutrition** – Assists dietitians and doctors in designing calorie-controlled diets.  

## Dataset and Features  
The dataset contains comprehensive physiological data, enabling precise calorie burn predictions.  

- **Total Records**: 15,000 entries  
- **Key Features**:  
  - **Activity Duration**  
  - **Heart Rate**  
  - **Body Temperature**  
  - **Age**  
  - **Height**  
  - **Weight**  

### Sample of the Dataset  

![01](https://github.com/user-attachments/assets/3742df7e-1995-4f0b-a055-aed646168d04)

*Sample of the Dataset* 

## Exploratory Data Analysis (EDA)  
Conducted **EDA** to uncover patterns, trends, and outliers using:  

- **Visualizations**  
- **Descriptive Statistics**  
- **Correlation Analysis** 

### Data Analysis Summary  

![02](https://github.com/user-attachments/assets/a2f2bdac-83f5-4d78-9f83-c217e54ca077)

*Data Analysis* 

### Age Distribution  
![03](https://github.com/user-attachments/assets/c7e4224c-8786-47ec-9f14-4b83b4424f27)

*Age Analysis* 

### Height Distribution  
![04](https://github.com/user-attachments/assets/0ff7f8e9-4fa4-49ac-8f8c-9f134248e594)

*Height Analysis* 

### Weight Distribution  
![05](https://github.com/user-attachments/assets/3daa10bd-32e0-45ac-98c2-30118eec729d)

*Weight Analysis* 

### Correlation Heatmap  
![06](https://github.com/user-attachments/assets/6890994a-ba03-4da9-ae7d-c12070ca380f)

*Correlation Heatmap* 

## Methodology  
This project followed a structured machine learning pipeline to achieve optimal performance:  

### **Data Preprocessing**  
- **Data Cleaning**: Removed missing values and outliers.  
- **Feature Engineering**: Created meaningful features for better prediction.  
- **Normalization**: Standardized features to improve model performance.  

### **Model Selection & Training**  
Evaluated multiple regression algorithms to identify the best-performing model.  

### Machine Learning Techniques Used  
- **Light Gradient Boosting Machine (LightGBM)**  
- **Extreme Gradient Boosting (XGBoost)**  
- **Extra Trees**  
- **Multi-layer Perceptron (MLP)**  
- **Random Forest**  
- **Gradient Boosting**  
- **Decision Tree**  
- **K-Nearest Neighbors (KNN)**  
- **Bayesian Ridge**  
- **Ridge Regression**  
- **Linear Regression**  
- **Huber Regression**  
- **Lasso Regression**  
- **ElasticNet**  
- **Support Vector Machine (SVM)**  

## Performance Comparison of All Techniques
The following graphs compare the Mean Absolute Error (MAE), R-Squared (R²), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for all models.

### Mean Absolute Error (MAE) Comparison
![07 1](https://github.com/user-attachments/assets/a798c74e-77cb-44ea-bf7e-a98b38520b2f)

*MAE Comparison* 

### R-Squared (R²) Comparison
![07 2](https://github.com/user-attachments/assets/4b18303a-11c2-4653-8407-936e94ad2c3d)

*R² Comparison* 

### Mean Squared Error (MSE) Comparison
![07 3](https://github.com/user-attachments/assets/8bf805f5-e47c-4375-a748-507188a20012)

*MSE Comparison* 

### Root Mean Squared Error (RMSE) Comparison
![07 4](https://github.com/user-attachments/assets/e7d3ca7f-218c-497f-b55a-55c49e581a6f)

*RMSE Comparison* 

## Code Snippets
Below are key sections of the code used in this project:

1. **LGBMRegressor Model Training**:
![08](https://github.com/user-attachments/assets/806da799-2d8a-4920-932d-799276a43db6)

2. **Plotting Graphs**:
![09](https://github.com/user-attachments/assets/50fc8001-7c9f-41ed-8dce-8d95c28b6b24)

## Conclusion
This project demonstrates how machine learning can effectively predict calories burned, delivering highly accurate results and valuable insights into activity-based calorie consumption.

### Key Takeaways  
- **Achieved state-of-the-art accuracy** with an **MAE score of 1.003**.  
- **Identified critical features** like heart rate and activity duration for precise predictions.  
- **Explored multiple ML techniques** to determine the optimal model.  
- **Developed insights for practical applications** in health, fitness, and sports.  

### Future Improvements  
- Enhance the dataset with **additional physiological factors**.  
- Optimize **hyperparameters** for even greater accuracy.  
- Deploy the model as an **API or web application**.  

## Explore the Repository
Explore the repository to learn more about the methodology, code, and results. Contributions and feedback are welcome!
