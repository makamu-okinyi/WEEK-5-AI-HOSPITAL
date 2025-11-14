"""
AI Development Workflow: Hospital Readmission Prediction System
Complete implementation from data preprocessing to deployment monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

#==============================================================================
# PART 1: DATA GENERATION (Simulating Hospital EHR Data)
#==============================================================================

def generate_synthetic_hospital_data(n_samples=5000):
    """
    Generate synthetic patient data mimicking EHR system
    Features include demographics, clinical measures, and social determinants
    """
    print("=" * 80)
    print("STEP 1: DATA GENERATION")
    print("=" * 80)
    
    # Demographics
    age = np.random.normal(65, 15, n_samples).clip(18, 95)
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.48, 0.52])
    
    # Clinical features
    num_comorbidities = np.random.poisson(2.5, n_samples)
    num_medications = np.random.poisson(5, n_samples)
    length_of_stay = np.random.gamma(2, 2, n_samples).clip(1, 30)
    
    # Lab values (with some missing data)
    hemoglobin = np.random.normal(13, 2, n_samples).clip(7, 18)
    creatinine = np.random.gamma(2, 0.5, n_samples).clip(0.5, 5)
    
    # Social determinants
    insurance = np.random.choice(['Medicare', 'Medicaid', 'Private', 'Uninsured'], 
                                 n_samples, p=[0.45, 0.20, 0.30, 0.05])
    distance_to_hospital = np.random.exponential(15, n_samples).clip(1, 100)
    
    # Prior utilization
    prior_admissions = np.random.poisson(1, n_samples)
    ed_visits_6mo = np.random.poisson(0.8, n_samples)
    
    # Diagnosis categories (simplified)
    diagnosis = np.random.choice(['Heart Failure', 'Pneumonia', 'COPD', 'Sepsis', 'Other'],
                                 n_samples, p=[0.25, 0.20, 0.15, 0.10, 0.30])
    
    # Discharge disposition
    discharge_location = np.random.choice(['Home', 'SNF', 'Rehab', 'Home Health'],
                                         n_samples, p=[0.60, 0.20, 0.10, 0.10])
    
    # Target: 30-day readmission (influenced by risk factors)
    risk_score = (
        0.02 * age +
        0.3 * num_comorbidities +
        0.2 * num_medications +
        0.1 * length_of_stay +
        0.4 * prior_admissions +
        0.3 * ed_visits_6mo +
        -0.5 * (hemoglobin - 13) +
        0.3 * creatinine +
        0.01 * distance_to_hospital +
        (insurance == 'Uninsured').astype(int) * 2 +
        (diagnosis == 'Heart Failure').astype(int) * 1.5 +
        np.random.normal(0, 2, n_samples)  # Random noise
    )
    
    # Convert risk score to probability and binary outcome
    prob_readmit = 1 / (1 + np.exp(-risk_score + 5))
    readmitted = (np.random.random(n_samples) < prob_readmit).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'num_comorbidities': num_comorbidities,
        'num_medications': num_medications,
        'length_of_stay': length_of_stay,
        'hemoglobin': hemoglobin,
        'creatinine': creatinine,
        'insurance': insurance,
        'distance_to_hospital': distance_to_hospital,
        'prior_admissions': prior_admissions,
        'ed_visits_6mo': ed_visits_6mo,
        'diagnosis': diagnosis,
        'discharge_location': discharge_location,
        'readmitted_30d': readmitted
    })
    
    # Introduce realistic missing data patterns
    missing_indices = np.random.choice(n_samples, int(0.15 * n_samples), replace=False)
    df.loc[missing_indices, 'hemoglobin'] = np.nan
    
    missing_indices = np.random.choice(n_samples, int(0.10 * n_samples), replace=False)
    df.loc[missing_indices, 'creatinine'] = np.nan
    
    print(f"✓ Generated {n_samples} synthetic patient records")
    print(f"✓ Readmission rate: {df['readmitted_30d'].mean():.1%}")
    print(f"✓ Features: {df.shape[1] - 1} (excluding target)")
    
    return df

#==============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
#==============================================================================

def perform_eda(df):
    """Comprehensive exploratory data analysis"""
    print("\n" + "=" * 80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    print("\n--- Data Overview ---")
    print(df.head())
    
    print("\n--- Data Types and Missing Values ---")
    print(df.info())
    
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    
    print("\n--- Target Distribution ---")
    print(df['readmitted_30d'].value_counts(normalize=True))
    
    print("\n--- Missing Data Analysis ---")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Visualizations
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Age distribution by readmission
    axes[0, 0].hist([df[df['readmitted_30d']==0]['age'], 
                     df[df['readmitted_30d']==1]['age']], 
                    bins=30, label=['Not Readmitted', 'Readmitted'], alpha=0.7)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Age Distribution by Readmission Status')
    axes[0, 0].legend()
    
    # Readmission by diagnosis
    diag_readmit = df.groupby('diagnosis')['readmitted_30d'].mean().sort_values()
    axes[0, 1].barh(diag_readmit.index, diag_readmit.values, color='steelblue')
    axes[0, 1].set_xlabel('Readmission Rate')
    axes[0, 1].set_title('Readmission Rate by Diagnosis')
    
    # Correlation heatmap (numeric features only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[1, 0], cbar_kws={'shrink': 0.8})
    axes[1, 0].set_title('Feature Correlation Matrix')
    
    # Number of comorbidities vs readmission
    comorbid_readmit = df.groupby('num_comorbidities')['readmitted_30d'].mean()
    axes[1, 1].plot(comorbid_readmit.index, comorbid_readmit.values, 
                    marker='o', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Comorbidities')
    axes[1, 1].set_ylabel('Readmission Rate')
    axes[1, 1].set_title('Readmission Rate by Comorbidity Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ EDA visualizations saved to 'eda_analysis.png'")

#==============================================================================
# PART 3: DATA PREPROCESSING
#==============================================================================

def preprocess_data(df):
    """
    Comprehensive preprocessing pipeline:
    - Handle missing values
    - Feature engineering
    - Encoding categorical variables
    - Scaling numeric features
    """
    print("\n" + "=" * 80)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 80)
    
    df_processed = df.copy()
    
    # 1. Feature Engineering
    print("\n--- Feature Engineering ---")
    
    # Create age groups
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                       bins=[0, 50, 65, 80, 100],
                                       labels=['<50', '50-65', '65-80', '80+'])
    
    # Polypharmacy flag
    df_processed['polypharmacy'] = (df_processed['num_medications'] >= 5).astype(int)
    
    # High-risk comorbidity flag
    df_processed['high_comorbidity'] = (df_processed['num_comorbidities'] >= 3).astype(int)
    
    # Interaction features
    df_processed['age_x_comorbid'] = df_processed['age'] * df_processed['num_comorbidities']
    
    # Days since last admission (simulated)
    df_processed['days_since_last_admit'] = np.where(
        df_processed['prior_admissions'] > 0,
        np.random.exponential(90, len(df_processed)),
        365  # No prior admission
    )
    
    # Log transform skewed features
    df_processed['log_length_of_stay'] = np.log1p(df_processed['length_of_stay'])
    df_processed['log_distance'] = np.log1p(df_processed['distance_to_hospital'])
    
    print("✓ Created 8 engineered features")
    
    # 2. Handle Missing Values
    print("\n--- Handling Missing Values ---")
    
    # Impute numeric features with median
    numeric_features = ['hemoglobin', 'creatinine']
    imputer = SimpleImputer(strategy='median')
    df_processed[numeric_features] = imputer.fit_transform(df_processed[numeric_features])
    
    # Create missing indicators
    for col in numeric_features:
        df_processed[f'{col}_missing'] = df[col].isnull().astype(int)
    
    print(f"✓ Imputed missing values in {len(numeric_features)} features")
    print(f"✓ Created {len(numeric_features)} missing indicators")
    
    # 3. Encode Categorical Variables
    print("\n--- Encoding Categorical Variables ---")
    
    # Binary encoding for gender
    df_processed['gender_encoded'] = (df_processed['gender'] == 'M').astype(int)
    
    # One-hot encoding for categorical features
    categorical_features = ['insurance', 'diagnosis', 'discharge_location', 'age_group']
    df_encoded = pd.get_dummies(df_processed, columns=categorical_features, 
                                 drop_first=True, dtype=int)
    
    print(f"✓ One-hot encoded {len(categorical_features)} categorical features")
    
    # 4. Separate features and target
    X = df_encoded.drop(['readmitted_30d', 'gender'], axis=1)
    y = df_encoded['readmitted_30d']
    
    print(f"\n✓ Final feature set: {X.shape[1]} features")
    print(f"✓ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

#==============================================================================
# PART 4: MODEL DEVELOPMENT & TRAINING
#==============================================================================

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and compare performance
    """
    print("\n" + "=" * 80)
    print("STEP 4: MODEL DEVELOPMENT & TRAINING")
    print("=" * 80)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    print("\n--- Handling Class Imbalance with SMOTE ---")
    if SMOTE is None:
        print("⚠️  SMOTE not available. Using original training set without balancing.")
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
    else:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"Original training set: {y_train.value_counts().to_dict()}")
    print(f"Balanced training set: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                         learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Train model
        model.fit(X_train_balanced, y_train_balanced)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation metrics
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"✓ Accuracy:  {results[name]['accuracy']:.4f}")
        print(f"✓ Precision: {results[name]['precision']:.4f}")
        print(f"✓ Recall:    {results[name]['recall']:.4f}")
        print(f"✓ F1-Score:  {results[name]['f1']:.4f}")
        print(f"✓ ROC-AUC:   {results[name]['roc_auc']:.4f}")
    
    return results, scaler

#==============================================================================
# PART 5: MODEL EVALUATION
#==============================================================================

def evaluate_models(results, y_test):
    """
    Comprehensive model evaluation with visualizations
    """
    print("\n" + "=" * 80)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 80)
    
    # Select best model (highest ROC-AUC)
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   ROC-AUC: {best_result['roc_auc']:.4f}")
    
    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, best_result['y_pred'])
    print(cm)
    
    # Detailed classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, best_result['y_pred'], 
                                target_names=['No Readmission', 'Readmission']))
    
    # Calculate specific metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"\nDetailed Metrics:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print(f"Specificity:     {tn/(tn+fp):.4f}")
    print(f"Sensitivity:     {tp/(tp+fn):.4f}")
    
    # Visualizations
    _, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['No Readmit', 'Readmit'],
                yticklabels=['No Readmit', 'Readmit'])
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_title(f'Confusion Matrix - {best_model_name}')
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, best_result['y_pred_proba'])
    axes[0, 1].plot(fpr, tpr, linewidth=2, 
                    label=f'ROC (AUC = {best_result["roc_auc"]:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, best_result['y_pred_proba'])
    axes[1, 0].plot(recall, precision, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Model Comparison
    model_names = list(results.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_plot):
        values = [results[name][metric] for name in model_names]
        axes[1, 1].bar(x + i*width, values, width, label=metric.upper())
    
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Model Performance Comparison')
    axes[1, 1].set_xticks(x + width * 2)
    axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Evaluation visualizations saved to 'model_evaluation.png'")
    
    return best_model_name, best_result

#==============================================================================
# PART 6: HYPERPARAMETER TUNING
#==============================================================================

def hyperparameter_tuning(X_train, y_train):
    """
    Grid search for optimal hyperparameters
    """
    print("\n" + "=" * 80)
    print("STEP 6: HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Balance classes
    if SMOTE is None:
        print("⚠️  SMOTE not available. Using original training set without balancing.")
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
    else:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("\n--- Grid Search for Random Forest ---")
    print(f"Parameter combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Perform grid search with cross-validation
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', 
                               n_jobs=-1, verbose=1)
    
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    print(f"\n✓ Best Parameters: {grid_search.best_params_}")
    print(f"✓ Best ROC-AUC Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

#==============================================================================
# PART 7: FAIRNESS ANALYSIS
#==============================================================================

def fairness_analysis(df, X_test, y_test, y_pred, y_pred_proba):
    """
    Analyze model fairness across demographic groups
    """
    print("\n" + "=" * 80)
    print("STEP 7: FAIRNESS & BIAS ANALYSIS")
    print("=" * 80)
    
    # Get original data for demographic analysis
    test_indices = X_test.index
    demographics = df.loc[test_indices, ['gender', 'insurance', 'age']].copy()
    demographics['y_true'] = y_test.values
    demographics['y_pred'] = y_pred
    demographics['y_pred_proba'] = y_pred_proba
    
    # Age stratification
    demographics['age_group'] = pd.cut(demographics['age'], 
                                       bins=[0, 50, 65, 80, 100],
                                       labels=['<50', '50-65', '65-80', '80+'])
    
    print("\n--- Fairness Metrics by Gender ---")
    for gender in demographics['gender'].unique():
        subset = demographics[demographics['gender'] == gender]
        print(f"\nGender: {gender}")
        print(f"  Sample size: {len(subset)}")
        print(f"  Precision:   {precision_score(subset['y_true'], subset['y_pred']):.4f}")
        print(f"  Recall:      {recall_score(subset['y_true'], subset['y_pred']):.4f}")
        print(f"  ROC-AUC:     {roc_auc_score(subset['y_true'], subset['y_pred_proba']):.4f}")
    
    print("\n--- Fairness Metrics by Insurance Type ---")
    for insurance in demographics['insurance'].unique():
        subset = demographics[demographics['insurance'] == insurance]
        if len(subset) > 10:  # Only analyze if sufficient samples
            print(f"\nInsurance: {insurance}")
            print(f"  Sample size: {len(subset)}")
            print(f"  Precision:   {precision_score(subset['y_true'], subset['y_pred']):.4f}")
            print(f"  Recall:      {recall_score(subset['y_true'], subset['y_pred']):.4f}")
            print(f"  ROC-AUC:     {roc_auc_score(subset['y_true'], subset['y_pred_proba']):.4f}")
    
    print("\n--- Fairness Metrics by Age Group ---")
    for age_grp in demographics['age_group'].unique():
        subset = demographics[demographics['age_group'] == age_grp]
        print(f"\nAge Group: {age_grp}")
        print(f"  Sample size: {len(subset)}")
        print(f"  Precision:   {precision_score(subset['y_true'], subset['y_pred']):.4f}")
        print(f"  Recall:      {recall_score(subset['y_true'], subset['y_pred']):.4f}")
        print(f"  ROC-AUC:     {roc_auc_score(subset['y_true'], subset['y_pred_proba']):.4f}")
    
    # Visualization
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Recall by insurance type
    recall_by_insurance = demographics.groupby('insurance').apply(
        lambda x: recall_score(x['y_true'], x['y_pred'])
    ).sort_values()
    
    axes[0].barh(recall_by_insurance.index, recall_by_insurance.values, color='coral')
    axes[0].set_xlabel('Recall (Sensitivity)')
    axes[0].set_title('Model Recall by Insurance Type')
    axes[0].axvline(recall_by_insurance.mean(), color='red', linestyle='--', 
                    label=f'Mean: {recall_by_insurance.mean():.3f}')
    axes[0].legend()
    
    # Recall by age group
    recall_by_age = demographics.groupby('age_group').apply(
        lambda x: recall_score(x['y_true'], x['y_pred'])
    )
    
    axes[1].bar(recall_by_age.index, recall_by_age.values, color='skyblue')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Recall (Sensitivity)')
    axes[1].set_title('Model Recall by Age Group')
    axes[1].axhline(recall_by_age.mean(), color='red', linestyle='--',
                    label=f'Mean: {recall_by_age.mean():.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('fairness_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Fairness analysis visualizations saved to 'fairness_analysis.png'")

#==============================================================================
# PART 8: DEPLOYMENT SIMULATION
#==============================================================================

def deployment_simulation(model, scaler, X_test, df):
    """
    Simulate model deployment with risk stratification
    """
    print("\n" + "=" * 80)
    print("STEP 8: DEPLOYMENT SIMULATION")
    print("=" * 80)
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Generate predictions
    predictions = model.predict_proba(X_test_scaled)[:, 1]
    
    # Risk stratification
    test_indices = X_test.index
    deployment_df = df.loc[test_indices].copy()
    deployment_df['risk_score'] = predictions
    deployment_df['risk_category'] = pd.cut(predictions, 
                                            bins=[0, 0.3, 0.6, 1.0],
                                            labels=['Low Risk', 'Moderate Risk', 'High Risk'])
    
    print("\n--- Risk Stratification ---")
    print(deployment_df['risk_category'].value_counts())
    
    print("\n--- Actual Readmission Rates by Risk Category ---")
    for category in ['Low Risk', 'Moderate Risk', 'High Risk']:
        subset = deployment_df[deployment_df['risk_category'] == category]
        if len(subset) > 0:
            actual_rate = subset['readmitted_30d'].mean()
            print(f"{category}: {actual_rate:.2%} ({len(subset)} patients)")
    
    # Intervention recommendations
    print("\n--- Recommended Interventions ---")
    high_risk = deployment_df[deployment_df['risk_category'] == 'High Risk']
    print(f"High Risk Patients: {len(high_risk)}")
    print("  → Schedule within 48 hours: Home health visit")
    print("  → Within 7 days: Primary care follow-up")
    print("  → Daily: Automated medication reminder calls")
    
    moderate_risk = deployment_df[deployment_df['risk_category'] == 'Moderate Risk']
    print(f"\nModerate Risk Patients: {len(moderate_risk)}")
    print("  → Within 14 days: Telehealth follow-up")
    print("  → Weekly: Symptom monitoring calls")
    
    # Sample high-risk patient profile
    print("\n--- Sample High-Risk Patient Profile ---")
    sample_patient = high_risk.iloc[0]
    print(f"Age: {sample_patient['age']:.0f}")
    print(f"Diagnosis: {sample_patient['diagnosis']}")
    print(f"Comorbidities: {sample_patient['num_comorbidities']:.0f}")
    print(f"Medications: {sample_patient['num_medications']:.0f}")
    print(f"Prior Admissions: {sample_patient['prior_admissions']:.0f}")
    print(f"Risk Score: {sample_patient['risk_score']:.2%}")
    
    return deployment_df

#==============================================================================
# PART 9: MONITORING & DRIFT DETECTION
#==============================================================================

def monitor_model_performance(deployment_df, window_size=100):
    """
    Simulate post-deployment monitoring and concept drift detection
    """
    print("\n" + "=" * 80)
    print("STEP 9: POST-DEPLOYMENT MONITORING")
    print("=" * 80)
    
    # Simulate time series of predictions
    deployment_df = deployment_df.sort_index()
    deployment_df['prediction_time'] = range(len(deployment_df))
    
    # Calculate rolling metrics
    deployment_df['rolling_accuracy'] = deployment_df.apply(
        lambda x: (x['risk_score'] > 0.5) == x['readmitted_30d'], axis=1
    ).rolling(window=window_size).mean()
    
    # Feature distribution monitoring
    print("\n--- Feature Distribution Monitoring ---")
    print(f"Mean age (current): {deployment_df['age'].mean():.1f}")
    print(f"Mean comorbidities (current): {deployment_df['num_comorbidities'].mean():.2f}")
    print(f"Mean length of stay (current): {deployment_df['length_of_stay'].mean():.2f}")
    
    # Simulate drift detection
    print("\n--- Concept Drift Detection ---")
    
    # Calculate prediction distribution drift
    early_predictions = deployment_df.iloc[:len(deployment_df)//2]['risk_score']
    late_predictions = deployment_df.iloc[len(deployment_df)//2:]['risk_score']
    
    from scipy import stats
    ks_statistic, p_value = stats.ks_2samp(early_predictions, late_predictions)
    
    print(f"KS Test Statistic: {ks_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("⚠️  WARNING: Significant distribution shift detected!")
        print("   Recommendation: Schedule model retraining")
    else:
        print("✓ No significant drift detected")
    
    # Performance degradation monitoring
    print("\n--- Performance Metrics Over Time ---")
    batches = np.array_split(deployment_df, 5)
    
    for i, batch in enumerate(batches):
        y_true = batch['readmitted_30d']
        y_pred = (batch['risk_score'] > 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        print(f"Batch {i+1}: Accuracy={accuracy:.4f}, Recall={recall:.4f}")
    
    # Visualization
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Rolling accuracy over time
    axes[0, 0].plot(deployment_df['prediction_time'], 
                    deployment_df['rolling_accuracy'], linewidth=2)
    axes[0, 0].set_xlabel('Prediction Number')
    axes[0, 0].set_ylabel(f'Rolling Accuracy (window={window_size})')
    axes[0, 0].set_title('Model Performance Over Time')
    axes[0, 0].axhline(y=0.75, color='r', linestyle='--', label='Target: 75%')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Risk score distribution over time
    axes[0, 1].hist([early_predictions, late_predictions], 
                    bins=30, label=['Early Predictions', 'Late Predictions'], 
                    alpha=0.7)
    axes[0, 1].set_xlabel('Risk Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Risk Score Distribution: Early vs Late')
    axes[0, 1].legend()
    
    # 3. Feature drift: Age distribution
    early_age = deployment_df.iloc[:len(deployment_df)//2]['age']
    late_age = deployment_df.iloc[len(deployment_df)//2:]['age']
    
    axes[1, 0].hist([early_age, late_age], bins=30, 
                    label=['Early Period', 'Late Period'], alpha=0.7)
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Feature Drift: Age Distribution')
    axes[1, 0].legend()
    
    # 4. Calibration plot
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        deployment_df['readmitted_30d'], 
        deployment_df['risk_score'], 
        n_bins=10
    )
    
    axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 
                    marker='o', linewidth=2, label='Model')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[1, 1].set_xlabel('Mean Predicted Probability')
    axes[1, 1].set_ylabel('Fraction of Positives')
    axes[1, 1].set_title('Calibration Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monitoring_drift.png', dpi=300, bbox_inches='tight')
    print("\n✓ Monitoring visualizations saved to 'monitoring_drift.png'")

#==============================================================================
# PART 10: FEATURE IMPORTANCE ANALYSIS
#==============================================================================

def feature_importance_analysis(model, X_train):
    """
    Analyze and visualize feature importance for model interpretability
    """
    print("\n" + "=" * 80)
    print("STEP 10: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_train.columns
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n--- Top 15 Most Important Features ---")
        print(importance_df.head(15).to_string(index=False))
        
        # Visualization
        _, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Top 15 features bar plot
        top_features = importance_df.head(15)
        axes[0].barh(range(len(top_features)), top_features['importance'])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['feature'])
        axes[0].set_xlabel('Importance Score')
        axes[0].set_title('Top 15 Feature Importances')
        axes[0].invert_yaxis()
        
        # Cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        axes[1].plot(range(len(importance_df)), 
                     importance_df['cumulative_importance'], 
                     linewidth=2, marker='o', markersize=3)
        axes[1].axhline(y=0.90, color='r', linestyle='--', 
                       label='90% Cumulative Importance')
        axes[1].set_xlabel('Number of Features')
        axes[1].set_ylabel('Cumulative Importance')
        axes[1].set_title('Cumulative Feature Importance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance visualizations saved to 'feature_importance.png'")
        
        return importance_df
    else:
        print("⚠️  Model does not support feature importance")
        return None

#==============================================================================
# PART 11: COST-BENEFIT ANALYSIS
#==============================================================================

def cost_benefit_analysis(y_test, y_pred, _y_pred_proba):
    """
    Analyze financial impact of model deployment
    Note: y_pred_proba parameter kept for API consistency but not currently used
    """
    print("\n" + "=" * 80)
    print("STEP 11: COST-BENEFIT ANALYSIS")
    print("=" * 80)
    
    # Cost assumptions (in dollars)
    READMISSION_COST = 15000  # Average cost of 30-day readmission
    INTERVENTION_COST = 500   # Cost of post-discharge intervention
    
    # Calculate confusion matrix
    # sklearn confusion_matrix returns [tn, fp, fn, tp] when raveled
    cm_result = confusion_matrix(y_test, y_pred).ravel()
    if len(cm_result) == 4:
        tn, fp, fn, tp = cm_result
    else:
        tn, fp, fn, tp = (0, 0, 0, 0)
    
    print("\n--- Financial Impact Analysis ---")
    
    # Without model (no interventions)
    total_readmissions_baseline = (tp + fn)
    baseline_cost = total_readmissions_baseline * READMISSION_COST
    
    print("\nBaseline (No Model):")
    print(f"  Total readmissions: {total_readmissions_baseline}")
    print(f"  Total cost: ${baseline_cost:,.2f}")
    
    # With model (assuming 50% intervention effectiveness)
    INTERVENTION_EFFECTIVENESS = 0.50
    prevented_readmissions = tp * INTERVENTION_EFFECTIVENESS
    remaining_readmissions = tp * (1 - INTERVENTION_EFFECTIVENESS) + fn
    
    intervention_costs = (tp + fp) * INTERVENTION_COST
    readmission_costs = remaining_readmissions * READMISSION_COST
    total_cost_with_model = intervention_costs + readmission_costs
    
    print("\nWith Model:")
    print(f"  Interventions provided: {tp + fp}")
    print(f"  Prevented readmissions: {prevented_readmissions:.0f}")
    print(f"  Remaining readmissions: {remaining_readmissions:.0f}")
    print(f"  Intervention costs: ${intervention_costs:,.2f}")
    print(f"  Readmission costs: ${readmission_costs:,.2f}")
    print(f"  Total cost: ${total_cost_with_model:,.2f}")
    
    # Net benefit
    net_savings = baseline_cost - total_cost_with_model
    roi = (net_savings / intervention_costs) * 100 if intervention_costs > 0 else 0
    
    print(f"\n💰 Net Savings: ${net_savings:,.2f}")
    print(f"📈 ROI: {roi:.1f}%")
    
    # Cost per patient
    cost_per_patient_baseline = baseline_cost / len(y_test)
    cost_per_patient_with_model = total_cost_with_model / len(y_test)
    
    print("\nPer Patient Analysis:")
    print(f"  Cost without model: ${cost_per_patient_baseline:,.2f}")
    print(f"  Cost with model: ${cost_per_patient_with_model:,.2f}")
    print(f"  Savings per patient: ${cost_per_patient_baseline - cost_per_patient_with_model:,.2f}")
    
    # Visualization
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cost comparison
    categories = ['Baseline\n(No Model)', 'With Model']
    costs = [baseline_cost, total_cost_with_model]
    colors = ['#ff6b6b', '#4ecdc4']
    
    bars = axes[0].bar(categories, costs, color=colors, alpha=0.7)
    axes[0].set_ylabel('Total Cost ($)')
    axes[0].set_title('Total Cost Comparison')
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'${cost:,.0f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Savings annotation
    axes[0].annotate('', xy=(1, baseline_cost), xytext=(1, total_cost_with_model),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    axes[0].text(1.15, (baseline_cost + total_cost_with_model)/2,
                f'Savings:\n${net_savings:,.0f}',
                fontsize=11, color='green', fontweight='bold')
    
    # Cost breakdown with model
    breakdown_labels = ['Interventions', 'Remaining\nReadmissions']
    breakdown_values = [intervention_costs, readmission_costs]
    breakdown_colors = ['#95e1d3', '#f38181']
    
    _, _, autotexts = axes[1].pie(breakdown_values, labels=breakdown_labels,
                                   colors=breakdown_colors, autopct='%1.1f%%',
                                   startangle=90)
    axes[1].set_title('Cost Breakdown With Model')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('cost_benefit_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Cost-benefit analysis saved to 'cost_benefit_analysis.png'")

#==============================================================================
# MAIN EXECUTION PIPELINE
#==============================================================================

def main():
    """
    Execute complete AI development workflow
    """
    print("\n" + "="*80)
    print(" "*20 + "HOSPITAL READMISSION PREDICTION SYSTEM")
    print(" "*25 + "Complete AI Workflow Pipeline")
    print("="*80)
    
    # Step 1: Generate synthetic data
    df = generate_synthetic_hospital_data(n_samples=5000)
    
    # Step 2: Exploratory Data Analysis
    perform_eda(df)
    
    # Step 3: Data Preprocessing
    X, y = preprocess_data(df)
    
    # Step 4: Train-Test Split (Time-aware)
    print("\n--- Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 5: Model Training
    results, scaler = train_models(X_train, X_test, y_train, y_test)
    
    # Step 6: Model Evaluation
    best_model_name, best_result = evaluate_models(results, y_test)
    print("\n--- Saving model artifacts ---")
    
    import joblib
    
    # 1. Save the best model
    # We use best_result['model'] which contains the trained model object
    best_model_object = results[best_model_name]['model']
    joblib.dump(best_model_object, 'readmission_model.pkl')
    print("✓ Model saved to 'readmission_model.pkl'")

    # 2. Save the scaler
    # The 'scaler' was created and fitted in your train_models() function
    joblib.dump(scaler, 'data_scaler.pkl')
    print("✓ Scaler saved to 'data_scaler.pkl'")
    
    # 3. Save the list of feature columns
    # This is CRITICAL for the API to process new data in the same order
    model_columns = X_train.columns
    joblib.dump(model_columns, 'model_columns.pkl')
    print("✓ Model columns saved to 'model_columns.pkl'")
    # -------------------------------------------------------------------

    
    # Step 7: Hyperparameter Tuning
    _tuned_model, _best_params = hyperparameter_tuning(X_train, y_train)
    
    # Step 8: Fairness Analysis
    fairness_analysis(df, X_test, y_test, 
                     best_result['y_pred'], 
                     best_result['y_pred_proba'])
    
    # Step 9: Feature Importance
    feature_importance_analysis(results[best_model_name]['model'], X_train)
    
    # Step 10: Deployment Simulation
    deployment_df = deployment_simulation(results[best_model_name]['model'], 
                                         scaler, X_test, df)
    
    # Step 11: Monitoring
    monitor_model_performance(deployment_df)
    
    # Step 12: Cost-Benefit Analysis
    cost_benefit_analysis(y_test, best_result['y_pred'], 
                         best_result['y_pred_proba'])
    
    # Final Summary
    print("\n" + "="*80)
    print(" "*30 + "WORKFLOW COMPLETE")
    print("="*80)
    print("\n📊 Generated Outputs:")
    print("  1. eda_analysis.png - Exploratory data analysis visualizations")
    print("  2. model_evaluation.png - Model performance metrics and comparisons")
    print("  3. fairness_analysis.png - Bias and fairness analysis across demographics")
    print("  4. feature_importance.png - Key predictive features")
    print("  5. monitoring_drift.png - Post-deployment monitoring and drift detection")
    print("  6. cost_benefit_analysis.png - Financial impact analysis")
    
    print("\n🎯 Key Findings:")
    print(f"  • Best Model: {best_model_name}")
    print(f"  • ROC-AUC Score: {best_result['roc_auc']:.4f}")
    print(f"  • Recall (Sensitivity): {best_result['recall']:.4f}")
    print(f"  • Precision: {best_result['precision']:.4f}")
    
    print("\n✅ Next Steps:")
    print("  1. Present findings to clinical stakeholders")
    print("  2. Conduct prospective validation study")
    print("  3. Develop integration plan with EHR system")
    print("  4. Establish monitoring dashboard")
    print("  5. Create clinician training materials")
    print("  6. Implement feedback loop for continuous improvement")
    
    print("\n" + "="*80 + "\n")

# Execute the pipeline
if __name__ == "__main__":
    main()