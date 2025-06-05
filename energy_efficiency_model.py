import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib
import os
import matplotlib.pyplot as plt

class EnergyEfficiencyModel:
    def __init__(self, model_dir='model_files'):
        """Initialize the Energy Efficiency Model."""
        self.model_dir = model_dir
        self.model = None
        self.le = None
        self.imputer = None
        self.scaler = None
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Feature order for consistency
        self.FEATURE_ORDER = [
            'Temperature (°C)',
            'Pressure (bar)',
            'CO2 Concentration (%)',
            'Flow Rate (m³/h)',
            'Sorbent Type',
            'Surface Area (m²/g)',
            'Pore Size (nm)',
            'Stability (1-10)',
            'pH Level',
            'CO2 Outlet Concentration (%)',
            'Energy Consumption (MJ/kg CO2)',
            'Cycle Time (min)',
            'Capture Efficiency (%)',
            'Regeneration Temp (°C)',
            'Solvent Concentration (%)',
            'Reactor Volume (m³)',
            'Residence Time (s)',
            'Reboiler Duty (kW)',
            'Column Height (m)'
        ]
        
        # Numerical columns (excluding 'Sorbent Type')
        self.NUMERICAL_COLS = [
            'Temperature (°C)',
            'Pressure (bar)',
            'CO2 Concentration (%)',
            'Flow Rate (m³/h)',
            'Surface Area (m²/g)',
            'Pore Size (nm)',
            'Stability (1-10)',
            'pH Level',
            'CO2 Outlet Concentration (%)',
            'Energy Consumption (MJ/kg CO2)',
            'Cycle Time (min)',
            'Capture Efficiency (%)',
            'Regeneration Temp (°C)',
            'Solvent Concentration (%)',
            'Reactor Volume (m³)',
            'Residence Time (s)',
            'Reboiler Duty (kW)',
            'Column Height (m)'
        ]

    def train(self, data_path):
        """Train the model using the provided dataset."""
        print("Loading dataset...")
        df = pd.read_csv(data_path)
        print(f"Dataset loaded successfully with {len(df)} rows")
        
        # Prepare features and targets
        target_columns = ['CO2 Emitted (kg/hr)', 'Energy Used Efficiently (MJ)']
        X = df.drop(columns=target_columns)
        y = df[target_columns]
        
        # Encode categorical variables
        self.le = LabelEncoder()
        X['Sorbent Type'] = self.le.fit_transform(X['Sorbent Type'])
        
        # Handle missing values and scale numerical features
        self.imputer = SimpleImputer(strategy='mean')
        X[self.NUMERICAL_COLS] = self.imputer.fit_transform(X[self.NUMERICAL_COLS])
        
        self.scaler = StandardScaler()
        X[self.NUMERICAL_COLS] = self.scaler.fit_transform(X[self.NUMERICAL_COLS])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
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
            n_jobs=-1
        )
        
        self.model = MultiOutputRegressor(xgb)
        
        # Perform cross-validation
        print("\nPerforming 5-fold cross-validation...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_co2 = []
        cv_scores_energy = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model_fold = MultiOutputRegressor(XGBRegressor(**xgb.get_params()))
            model_fold.fit(X_fold_train, y_fold_train)
            
            y_fold_pred = model_fold.predict(X_fold_val)
            
            r2_co2 = r2_score(y_fold_val['CO2 Emitted (kg/hr)'], y_fold_pred[:, 0])
            r2_energy = r2_score(y_fold_val['Energy Used Efficiently (MJ)'], y_fold_pred[:, 1])
            
            cv_scores_co2.append(r2_co2)
            cv_scores_energy.append(r2_energy)
            
            print(f"Fold {fold} - CO2 R²: {r2_co2:.4f}, Energy R²: {r2_energy:.4f}")
        
        # Train final model
        print("\nTraining final model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        print("\nFinal Model Performance Metrics:")
        print("-" * 50)
        overall_r2 = 0
        for i, target in enumerate(target_columns):
            mae = mean_absolute_error(y_test[target], y_pred[:, i])
            mse = mean_squared_error(y_test[target], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test[target], y_pred[:, i])
            overall_r2 += r2
            
            print(f"\nMetrics for {target}:")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R² Score: {r2:.4f}")
        
        overall_accuracy = (overall_r2 / 2) * 100
        print(f"\nOverall Model Accuracy: {overall_accuracy:.2f}%")
        
        # Save model and preprocessors
        self.save_model()
        
        # Plot learning curves
        self.plot_learning_curves(X_train, y_train)

    def save_model(self):
        """Save the trained model and preprocessing objects."""
        print("\nSaving model and preprocessing objects...")
        joblib.dump(self.model, os.path.join(self.model_dir, 'carbon_emission_model.joblib'))
        joblib.dump(self.le, os.path.join(self.model_dir, 'label_encoder.joblib'))
        joblib.dump(self.imputer, os.path.join(self.model_dir, 'imputer.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))
        print("Model and preprocessing objects saved successfully!")

    def load_model(self):
        """Load the trained model and preprocessing objects."""
        self.model = joblib.load(os.path.join(self.model_dir, 'carbon_emission_model.joblib'))
        self.le = joblib.load(os.path.join(self.model_dir, 'label_encoder.joblib'))
        self.imputer = joblib.load(os.path.join(self.model_dir, 'imputer.joblib'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
        print("Model and preprocessing objects loaded successfully!")

    def preprocess_input(self, input_data):
        """Preprocess input data for prediction."""
        df_input = pd.DataFrame([input_data])
        df_input = df_input[self.FEATURE_ORDER]
        df_input['Sorbent Type'] = self.le.transform(df_input['Sorbent Type'])
        df_input[self.NUMERICAL_COLS] = self.imputer.transform(df_input[self.NUMERICAL_COLS])
        df_input[self.NUMERICAL_COLS] = self.scaler.transform(df_input[self.NUMERICAL_COLS])
        return df_input

    def predict(self, input_data):
        """Make predictions using the trained model."""
        if self.model is None:
            self.load_model()
        
        processed_data = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_data)
        
        return {
            'CO2 Emitted (kg/hr)': prediction[0][0],
            'Energy Used Efficiently (MJ)': prediction[0][1]
        }

    def analyze_parameter_sensitivity(self, input_data, variation_percentage=10):
        """Analyze parameter sensitivity and provide optimization recommendations."""
        base_prediction = self.predict(input_data)
        base_co2 = base_prediction['CO2 Emitted (kg/hr)']
        base_energy = base_prediction['Energy Used Efficiently (MJ)']
        
        recommendations = {
            'increase': [],
            'decrease': [],
            'neutral': []
        }
        
        for param in self.NUMERICAL_COLS:
            increased_data = input_data.copy()
            decreased_data = input_data.copy()
            
            current_value = input_data[param]
            variation = current_value * (variation_percentage / 100)
            
            increased_data[param] = current_value + variation
            decreased_data[param] = current_value - variation
            
            increased_pred = self.predict(increased_data)
            decreased_pred = self.predict(decreased_data)
            
            co2_increase_impact = increased_pred['CO2 Emitted (kg/hr)'] - base_co2
            co2_decrease_impact = decreased_pred['CO2 Emitted (kg/hr)'] - base_co2
            energy_increase_impact = increased_pred['Energy Used Efficiently (MJ)'] - base_energy
            energy_decrease_impact = decreased_pred['Energy Used Efficiently (MJ)'] - base_energy
            
            if co2_increase_impact < 0 and energy_increase_impact > 0:
                recommendations['increase'].append({
                    'parameter': param,
                    'co2_impact': abs(co2_increase_impact),
                    'energy_impact': energy_increase_impact
                })
            elif co2_decrease_impact < 0 and energy_decrease_impact > 0:
                recommendations['decrease'].append({
                    'parameter': param,
                    'co2_impact': abs(co2_decrease_impact),
                    'energy_impact': energy_decrease_impact
                })
            else:
                recommendations['neutral'].append({
                    'parameter': param,
                    'co2_impact': min(abs(co2_increase_impact), abs(co2_decrease_impact)),
                    'energy_impact': max(energy_increase_impact, energy_decrease_impact)
                })
        
        return recommendations

    def print_optimization_recommendations(self, recommendations, input_data):
        """Print formatted optimization recommendations."""
        print("\nOptimization Recommendations for Better Performance:")
        print("-" * 50)
        
        if recommendations['increase']:
            print("\nINCREASE these parameters:")
            for rec in sorted(recommendations['increase'], key=lambda x: x['energy_impact'], reverse=True):
                print(f"• {rec['parameter']} (Current: {input_data[rec['parameter']]})")
                print(f"  Expected improvement: {rec['co2_impact']:.2f} kg/hr less CO2, {rec['energy_impact']:.2f} MJ more efficiency")
        
        if recommendations['decrease']:
            print("\nDECREASE these parameters:")
            for rec in sorted(recommendations['decrease'], key=lambda x: x['energy_impact'], reverse=True):
                print(f"• {rec['parameter']} (Current: {input_data[rec['parameter']]})")
                print(f"  Expected improvement: {rec['co2_impact']:.2f} kg/hr less CO2, {rec['energy_impact']:.2f} MJ more efficiency")
        
        if not recommendations['increase'] and not recommendations['decrease']:
            print("\nNo significant improvements found. Current parameters are well optimized.")

    def plot_learning_curves(self, X, y):
        """Plot learning curves for the model."""
        plt.figure(figsize=(10, 6))
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        
        plt.xlabel('Training Examples')
        plt.ylabel('R² Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(self.model_dir, 'learning_curves.png'))
        plt.close()

def main():
    # Create model instance
    model = EnergyEfficiencyModel()
    
    # Train the model with your dataset
    model.train("dataset_path")
    
    # Example input for prediction
    sample_input = {
        'Temperature (°C)': 85,
        'Pressure (bar)': 5,
        'Flow Rate (m³/h)': 1000,
        'CO2 Concentration (%)': 12,
        'CO2 Outlet Concentration (%)': 2,
        'Capture Efficiency (%)': 85,
        'Column Height (m)': 10,
        'Cycle Time (min)': 30,
        'Energy Consumption (MJ/kg CO2)': 2.5,
        'Pore Size (nm)': 50,
        'Reactor Volume (m³)': 5,
        'Reboiler Duty (kW)': 100,
        'Regeneration Temp (°C)': 120,
        'Residence Time (s)': 60,
        'Solvent Concentration (%)': 30,
        'Stability (1-10)': 8,
        'Surface Area (m²/g)': 200,
        'pH Level': 7,
        'Sorbent Type': 'MEA'
    }
    
    # Make prediction
    result = model.predict(sample_input)
    print("\nCurrent Prediction Results:")
    print(f"Predicted CO2 Emitted: {result['CO2 Emitted (kg/hr)']:.2f} kg/hr")
    print(f"Predicted Energy Used Efficiently: {result['Energy Used Efficiently (MJ)']:.2f} MJ")
    
    # Get optimization recommendations
    recommendations = model.analyze_parameter_sensitivity(sample_input)
    model.print_optimization_recommendations(recommendations, sample_input)

if __name__ == "__main__":
    main() 
