import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

# Load the trained model and scaler
with open('EvaluateRiskModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('EvaluateRiskModelScaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the categorical feature mappings used during training
reason_categories = ["Other reason", "Home ownership", "Not home ownership"]
job_categories = ["Mgr", "Other job", "Office worker", "Retail worker", "Business owner", "Technician"]

# Function to make predictions
def make_prediction():
    try:
        # Get values from input fields
        loan = float(entry_loan.get())
        mortdue = float(entry_mortdue.get())
        value = float(entry_value.get())
        reason = var_reason.get()
        job = var_job.get()
        yoj = float(entry_yoj.get())
        derog = float(entry_derog.get())
        delinq = float(entry_delinq.get())
        clage = float(entry_clage.get())
        ninq = float(entry_ninq.get())
        clno = float(entry_clno.get())
        debtinc = float(entry_debtinc.get())
        
        # Prepare the feature vector
        features = np.array([[loan, mortdue, value, yoj, derog, delinq, clage, ninq, clno, debtinc,0]])

        # One-hot encode the categorical columns (Reason and Job) based on predefined categories
        reason_encoded = [1 if reason == r else 0 for r in reason_categories]
        job_encoded = [1 if job == j else 0 for j in job_categories]
        
        # Add the encoded values to the feature vector
        features = np.concatenate([features, [reason_encoded + job_encoded]], axis=1)

        # Preprocess the data (same as during training)
        print("features", features)
        features_scaled = scaler.transform(features)
        
        # Make prediction using the model
        prediction = model.predict(features_scaled)
        
        # Display result
        if prediction == 1:
            messagebox.showinfo("Prediction", "The customer is likely to default on the loan.")
        else:
            messagebox.showinfo("Prediction", "The customer is likely to repay the loan.")
    
    except Exception as e:
        messagebox.showerror("Input Error", f"An error occurred: {str(e)}")

# Create the main window
root = tk.Tk()
root.title("Loan Default Prediction")

# Create and place labels and entry widgets for each feature with default values
tk.Label(root, text="Loan Amount").grid(row=0, column=0, padx=10, pady=5)
entry_loan = tk.Entry(root)
entry_loan.insert(0, "10000")  # Default value
entry_loan.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Mortgage Due").grid(row=1, column=0, padx=10, pady=5)
entry_mortdue = tk.Entry(root)
entry_mortdue.insert(0, "5000")  # Default value
entry_mortdue.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Value").grid(row=2, column=0, padx=10, pady=5)
entry_value = tk.Entry(root)
entry_value.insert(0, "20000")  # Default value
entry_value.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Reason").grid(row=3, column=0, padx=10, pady=5)
var_reason = tk.StringVar(value="Home ownership")  # Default value
reason_options = reason_categories
dropdown_reason = tk.OptionMenu(root, var_reason, *reason_options)
dropdown_reason.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Job").grid(row=4, column=0, padx=10, pady=5)
var_job = tk.StringVar(value="Office worker")  # Default value
job_options = job_categories
dropdown_job = tk.OptionMenu(root, var_job, *job_options)
dropdown_job.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Years of Job").grid(row=5, column=0, padx=10, pady=5)
entry_yoj = tk.Entry(root)
entry_yoj.insert(0, "5")  # Default value
entry_yoj.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Derogatory Events").grid(row=6, column=0, padx=10, pady=5)
entry_derog = tk.Entry(root)
entry_derog.insert(0, "0")  # Default value
entry_derog.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Delinquencies").grid(row=7, column=0, padx=10, pady=5)
entry_delinq = tk.Entry(root)
entry_delinq.insert(0, "0")  # Default value
entry_delinq.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Current Credit Balance").grid(row=8, column=0, padx=10, pady=5)
entry_clage = tk.Entry(root)
entry_clage.insert(0, "1000")  # Default value
entry_clage.grid(row=8, column=1, padx=10, pady=5)

tk.Label(root, text="Number of Inquiries").grid(row=9, column=0, padx=10, pady=5)
entry_ninq = tk.Entry(root)
entry_ninq.insert(0, "2")  # Default value
entry_ninq.grid(row=9, column=1, padx=10, pady=5)

tk.Label(root, text="Number of Credit Lines").grid(row=10, column=0, padx=10, pady=5)
entry_clno = tk.Entry(root)
entry_clno.insert(0, "5")  # Default value
entry_clno.grid(row=10, column=1, padx=10, pady=5)

tk.Label(root, text="Debt-to-Income Ratio").grid(row=11, column=0, padx=10, pady=5)
entry_debtinc = tk.Entry(root)
entry_debtinc.insert(0, "30")  # Default value
entry_debtinc.grid(row=11, column=1, padx=10, pady=5)

# Create a button to make the prediction
predict_button = tk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=12, column=0, columnspan=2, pady=10)

# Run the application
root.mainloop()
