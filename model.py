import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import sys
import warnings
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

try:
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(r"C:\Users\vinays\Downloads\carbon_capture_dataset_2000_extended\carbon_capture_dataset_2000_extended.csv")
    print(f"Dataset loaded successfully with {len(df)} rows")
    
    # Print column names to verify
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    
    # Features and targets
    print("\nPreparing features and targets...")
    # Use correct target column names from dataset
    target_columns = ['CO2 Emitted (kg/hr)', 'Energy Used Efficiently (MJ)']
    if not all(col in df.columns for col in target_columns):
        print("Warning: Target columns not found with exact names. Please check the column names above.")
        sys.exit(1)
    X = df.drop(columns=target_columns)
    y = df[target_columns]

    # Print required feature columns for prediction
    print("\nRequired feature columns for prediction:")
    print(X.columns.tolist())
    # Print exact order of feature columns for debugging
    print("\nExact order of feature columns used during training:")
    print(X.columns.tolist())

    # Encode categorical variables
    le = LabelEncoder()
    X['Sorbent Type'] = le.fit_transform(X['Sorbent Type'])

    # Identify numerical columns (excluding categorical)
    numerical_cols = [col for col in X.columns if X[col].dtype in [np.float64, np.int64] and col != 'Sorbent Type']
    imputer = SimpleImputer(strategy='mean')
    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print("Features prepared and scaled successfully")

    # Train-test split
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Initialize base regressor with optimized hyperparameters
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1  # Use all available cores
    )

    # Wrap in MultiOutputRegressor
    model = MultiOutputRegressor(xgb)

    # Perform k-fold cross-validation
    print("\nPerforming 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_co2 = []
    cv_scores_energy = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        # Split data
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model
        model_fold = MultiOutputRegressor(XGBRegressor(**xgb.get_params()))
        model_fold.fit(X_fold_train, y_fold_train)
        
        # Predict
        y_fold_pred = model_fold.predict(X_fold_val)
        
        # Calculate scores
        r2_co2 = r2_score(y_fold_val['CO2 Emitted (kg/hr)'], y_fold_pred[:, 0])
        r2_energy = r2_score(y_fold_val['Energy Used Efficiently (MJ)'], y_fold_pred[:, 1])
        
        cv_scores_co2.append(r2_co2)
        cv_scores_energy.append(r2_energy)
        
        print(f"Fold {fold} - CO2 R²: {r2_co2:.4f}, Energy R²: {r2_energy:.4f}")

    print(f"\nCross-validation results:")
    print(f"CO2 R² mean: {np.mean(cv_scores_co2):.4f} (±{np.std(cv_scores_co2):.4f})")
    print(f"Energy R² mean: {np.mean(cv_scores_energy):.4f} (±{np.std(cv_scores_energy):.4f})")

    # Train final model on full training set
    print("\nTraining final model...")
    model.fit(X_train, y_train)

    # Make predictions
    print("\nMaking predictions and calculating metrics...")
    y_pred = model.predict(X_test)

    # Calculate detailed metrics
    print("\nFinal Model Performance Metrics:")
    print("-" * 50)
    overall_r2 = 0
    for i, target in enumerate(['CO2 Emitted (kg/hr)', 'Energy Used Efficiently (MJ)']):
        mae = mean_absolute_error(y_test[target], y_pred[:, i])
        mse = mean_squared_error(y_test[target], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test[target], y_pred[:, i])
        overall_r2 += r2
        
        print(f"\nMetrics for {target}:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score (Accuracy): {r2:.4f}")
        print(f"Mean Actual Value: {y_test[target].mean():.4f}")
        print(f"Mean Predicted Value: {y_pred[:, i].mean():.4f}")
    
    # Print overall model accuracy scaled to 100%
    overall_accuracy = (overall_r2 / 2) * 100
    print(f"\nOverall Model Accuracy: {overall_accuracy:.2f}%")

    # Feature importance analysis
    print("\nFeature Importance Analysis:")
    print("-" * 50)
    feature_names = X.columns
    for i, target in enumerate(['CO2 Emitted (kg/hr)', 'Energy Used Efficiently (MJ)']):
        importances = model.estimators_[i].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nTop features for {target}:")
        for f in range(len(feature_names)):
            print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

    # Save the model and preprocessing objects
    print("\nSaving model and preprocessing objects...")
    joblib.dump(model, 'carbon_emission_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    joblib.dump(imputer, 'imputer.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and preprocessing objects have been saved successfully!")

    # Plot learning curves
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.show()

    # Plot learning curves for the model
    plot_learning_curve(model, "Learning Curves (XGBoost)", X_train, y_train)

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print(f"Error type: {type(e)._name_}")
    sys.exit(1)

try:
    # Load model and preprocessing objects
    print("\nLoading model and preprocessing objects...")
    try:
        model = joblib.load('carbon_emission_model.joblib')
        le = joblib.load('label_encoder.joblib')
        imputer = joblib.load('imputer.joblib')
        scaler = joblib.load('scaler.joblib')
        print("Model files loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: Could not find model files. Please make sure you have run the training part of the code first.")
        print(f"Missing file: {str(e)}")
        sys.exit(1)

    # Example input (make sure feature order matches training)
    sample = {col: 0 for col in X.columns}
    # Provide reasonable defaults for a few known columns
    sample['Temperature (°C)'] = 85
    sample['Pressure (bar)'] = 5
    sample['Flow Rate (m³/h)'] = 1000
    sample['CO2 Concentration (%)'] = 12
    sample['Sorbent Type'] = 'MEA'
    # You can update other values as needed

    # Create DataFrame and preprocess
    try:
        df_sample = pd.DataFrame([sample])
        df_sample['Sorbent Type'] = le.transform(df_sample['Sorbent Type'])
        # Scale numerical features
        df_sample[numerical_cols] = imputer.transform(df_sample[numerical_cols])
        df_sample[numerical_cols] = scaler.transform(df_sample[numerical_cols])
        # Make prediction
        pred = model.predict(df_sample)
        print(f"\nPrediction Results:")
        print(f"Predicted CO2 Emitted: {pred[0][0]:.2f} kg/hr")
        print(f"Predicted Energy Used Efficiently: {pred[0][1]:.2f} MJ")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("Please make sure all required features are present and in the correct format.")
        sys.exit(1)

except Exception as e:
    print(f"\nUnexpected error: {str(e)}")
    sys.exit(1)
    
  