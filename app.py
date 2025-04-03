from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and encoder
model = None
gender_encoder = None

# Load and preprocess the dataset
def preprocess_data():
    df = pd.read_csv('Disease_symptom_and_patient_profile_dataset1.csv')

    # Convert categorical variables to numerical
    le_gender = LabelEncoder()
    df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])

    # Define symptom columns
    symptom_columns = ['Fever', 'Cough', 'Fatigue', 'Shortness of Breath', 'Chest Pain', 'Headache',
                      'Nausea', 'Joint Pain', 'Sore Throat', 'Runny Nose', 'Sneezing', 'Abdominal Pain',
                      'Skin Rash', 'Frequent Urination', 'Back Pain', 'Weight Loss', 'Night Sweats',
                      'Chills', 'Loss of Taste', 'Difficulty Swallowing']

    # Convert Yes/No to 1/0 for symptoms
    for col in symptom_columns:
        df[col] = (df[col] == 'Yes').astype(int)

    # Prepare features and target
    X = df[['Age', 'Gender_encoded'] + symptom_columns]
    y = df['Disease']

    # Ensure a "No Disease" category for cases with no symptoms
    no_symptom_mask = X[symptom_columns].sum(axis=1) == 0
    if "No Disease" not in y.unique():
        df.loc[no_symptom_mask, 'Disease'] = "No Disease"

    # Convert to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()

    # Reduce noise factor to avoid unwanted artifacts
    noise_factor = 0.35
    num_samples = X.shape[0]
    num_features = X.shape[1]

    # Add slight random noise to features
    noise = np.random.normal(0, noise_factor, (num_samples, num_features))
    X_noisy = X + noise

    # Data augmentation: create synthetic samples
    num_synthetic = int(num_samples * 0.3)  
    synthetic_indices = np.random.choice(num_samples, size=(num_synthetic, 2))

    for i, j in synthetic_indices:
        mix_ratio = np.random.beta(0.4, 0.4)
        new_features = X[i] * mix_ratio + X[j] * (1 - mix_ratio)
        new_features += np.random.normal(0, noise_factor, num_features)

        # Append synthetic sample
        X_noisy = np.vstack([X_noisy, new_features])

        # Randomly choose one of the parent labels
        new_label = y[i] if np.random.random() < mix_ratio else y[j]
        y = np.append(y, new_label)

    # Perform k-fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []

    print("Cross-validation Results:")
    fold = 1

    for train_index, test_index in kf.split(X_noisy):
        X_train, X_test = X_noisy[train_index], X_noisy[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate model
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores.append(accuracy)

        print(f"\nFold {fold} Results:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        fold += 1

    # Calculate and print average metrics
    mean_accuracy = np.mean(cv_scores)
    print("\nOverall Model Performance:")
    print(f"Mean Accuracy: {mean_accuracy:.2f}")

    # Train final model on full dataset
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_noisy, y)

    # Save the model and encoders
    joblib.dump(rf_model, 'disease_model.joblib')
    joblib.dump(le_gender, 'gender_encoder.joblib')

    return rf_model, le_gender

def init_app():
    global model, gender_encoder
    model, gender_encoder = preprocess_data()

# Initialize the model
init_app()

def create_visualizations():
    df = pd.read_csv('Disease_symptom_and_patient_profile_dataset1.csv')
    plots = {}
    
    # Set style for all plots
    plt.style.use('default')
    sns.set_style('whitegrid')
    # Set the color palette
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6']
    sns.set_palette(colors)
    
    try:
        # 1. Disease Distribution
        plt.figure(figsize=(12, 6))
        disease_counts = df['Disease'].value_counts()
        sns.barplot(x=disease_counts.values, y=disease_counts.index)
        plt.title('Disease Distribution')
        plt.xlabel('Number of Cases')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        plots['disease_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # 2. Age Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Age', bins=30)
        plt.title('Age Distribution')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        plots['age_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # 3. Gender Distribution
        plt.figure(figsize=(8, 6))
        df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Gender Distribution')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        plots['gender_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # 4. Symptom Frequency
        symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Shortness of Breath', 'Chest Pain', 'Headache',
                        'Nausea', 'Joint Pain', 'Sore Throat', 'Runny Nose', 'Sneezing', 'Abdominal Pain',
                        'Skin Rash', 'Frequent Urination', 'Back Pain', 'Weight Loss', 'Night Sweats',
                        'Chills', 'Loss of Taste', 'Difficulty Swallowing']
        
        symptom_freq = df[symptom_cols].apply(lambda x: (x == 'Yes').sum()).sort_values(ascending=True)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=symptom_freq.values, y=symptom_freq.index)
        plt.title('Symptom Frequency')
        plt.xlabel('Number of Cases')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        plots['symptom_freq'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # 5. Symptom Correlation
        plt.figure(figsize=(15, 12))
        symptom_corr = df[symptom_cols].apply(lambda x: pd.Series(x == 'Yes', dtype=int))
        sns.heatmap(symptom_corr.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Symptom Correlation Matrix')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.5)
        plt.close()
        plots['symptom_corr'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error in create_visualizations: {str(e)}")
        return {}
    
    return plots

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    df = pd.read_csv('Disease_symptom_and_patient_profile_dataset1.csv')
    unique_genders = sorted(df['Gender'].unique())
    symptoms = ['Fever', 'Cough', 'Fatigue', 'Shortness of Breath', 'Chest Pain', 'Headache',
               'Nausea', 'Joint Pain', 'Sore Throat', 'Runny Nose', 'Sneezing', 'Abdominal Pain',
               'Skin Rash', 'Frequent Urination', 'Back Pain', 'Weight Loss', 'Night Sweats',
               'Chills', 'Loss of Taste', 'Difficulty Swallowing']
    
    # Generate plots
    plots = create_visualizations()
    
    return render_template('index.html', 
                           unique_genders=unique_genders, 
                           symptoms=symptoms,
                           plots=plots)

@app.route('/visualize')
def visualize():
    # Generate visualizations
    plots = create_visualizations()
    return render_template('visualize.html', plots=plots)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        gender = request.form['gender']

        symptoms = ['Fever', 'Cough', 'Fatigue', 'Shortness of Breath', 'Chest Pain', 'Headache',
                   'Nausea', 'Joint Pain', 'Sore Throat', 'Runny Nose', 'Sneezing', 'Abdominal Pain',
                   'Skin Rash', 'Frequent Urination', 'Back Pain', 'Weight Loss', 'Night Sweats',
                   'Chills', 'Loss of Taste', 'Difficulty Swallowing']

        # Create feature vector
        features = []
        features.append(age)
        features.append(gender_encoder.transform([gender])[0])

        # Add symptom values (1 for Yes, 0 for No)
        for symptom in symptoms:
            features.append(1 if request.form.get(symptom) == 'Yes' else 0)

        # Check if all symptoms are 'no'
        symptom_values = features[2:]  # Exclude age and gender
        if all(sv == 0 for sv in symptom_values):
            main_prediction = "No Disease"
            results = []
        else:
            # Make prediction only if there are symptoms
            features = np.array(features).reshape(1, -1)
            probabilities = model.predict_proba(features)[0]

            # Get top 3 most likely diseases
            disease_probs = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            top_disease, top_prob = disease_probs[0]

            # Prevent forced predictions with a threshold
            if top_prob < 0.2:  # If no disease has >20% probability
                main_prediction = "No Disease"
                results = []
            else:
                main_prediction = top_disease
                results = [{'disease': disease, 'probability': f"{prob*100:.1f}%"} for disease, prob in disease_probs[:3] if prob > 0.1]

        return render_template('result.html', main_prediction=main_prediction, results=results)

    except Exception as e:
        return render_template('result.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5007)
