Shipment Delay Prediction System

## Most Important- Note
Dataset: https://drive.google.com/drive/folders/1I-h-HZPXzQZzOU6iF7YrS8XLQD1NcbeW  
                                 OR 
         https://data.mendeley.com/datasets/8gx2fvg2k6/5 - Use this link and download file one with 91.5 MB
             Download it and place it in same directory as final_project.ipynb

 Outline:   
   * This project uses Data loading and Preprocessing, Feature Engineering, Data     Analysis, SQL-based querying, Machine Learning, and Model Training to investigate Shipment delays in e-commerce supply chains. 
   * The main goal is to predict shipment delay of items ordered using trends linked to shipping delay and to predict whether a shipment will be delayed or not by creating prediction models. 
   * A Streamlit web application was also created to see real-time shipment delay prediction. 


 Requirements to run the code: 
 - Python
 - Pandas
 - NumPy
 - Matplotlib
 - Seaborn
 - Scikit-learn
 - Streamlit
 - Pickle
 - SQL

 This Project covers: 
 - Data loading, Data cleaning, and Data preprocessing
 - Feature Engineering
 - Data Analysis
 - SQL-based Queries
 - Machine Learning: 1. Logistic Regression, 2. Decision Tree, 3. Random Forest
 - Model Evaluation using: 1. Accuracy, 2. ROC-AUC, 3. Confusion Matrix, 
                           4.  Cross Validation
 - Feature Importance Analysis
 - Streamlit App for Live prediction

 Project Files:
 /Final Project/
 -- final_project.ipynb       ## Main Notebook
 -- app.py                    ## Code for App
 -- dataco_supply_chain.csv   ## Data set
 -- rf_model.pkl              ## Random Forest Model
 -- model_columns.pkl         ## Feature columns for prediction
 -- requirements.txt          ## Required Libraries
 -- README.md                 ## Project documentation

 How to Run the Notebook: 
 1. Open 'final_project.ipynb' on codebench
 2. Download 'dataco_supply_chain.csv' and put it in Codebench in the same directory
 3. Reload or restart the kernel and run all the cells 

 How to Run the Web App: 
 1. Install:  pip install -r requirements.txt
 2. Run App: streamlit run app.py

 Model Performance

          Model                Accuracy             AUC
      Logistic Regression       0.700               0.746
      Decision Tree             0.780               0.775
      Random Forest             0.724               0.803

      * Random Forest performed the best overall, and we selected Random Forest as our ultimate Model as it has the highest AUC. 
    
 Takeaways
   * Faster shipping choices have greater delay rates.
   * Delay patterns are consistent across all regions and time periods.
   * Machine Model predicted shipment delays using operational and categorical features.

 Streamlit Application
   * App allows users to predict shipment delay status by entering shipment details like region, and so on, and get the percentage of how much it will be delayed.
Note: rf_model.pkl and model_columns.pkl are generated automatically when you run all cells in codebench (Jupyter file) - final_project.ipynb. 



