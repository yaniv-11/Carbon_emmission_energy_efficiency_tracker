# app.py
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from model import EnergyEfficiencyModel
import os
import traceback
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key' # Add a secret key for sessions

# Initialize model
try:
    logger.info("Initializing EnergyEfficiencyModel...")
    model = EnergyEfficiencyModel()
    model_path = os.path.join('model_files', 'carbon_emission_model.joblib')
    
    if not os.path.exists(model_path):
        logger.info("Model file not found. Training new model...")
        dataset_path = ""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        model.train(dataset_path)
    else:
        logger.info("Loading existing model...")
        model.load_model()
    logger.info("Model initialization completed successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/app')
def app_page():
    try:
        # Load the model if not already loaded
        if model.le is None:
            model.load_model()
            
        # Get the list of sorbent types from the LabelEncoder classes
        sorbent_types = list(model.le.classes_)
        logger.info(f"Available sorbent types: {sorbent_types}")
        
        # Render the index.html template and pass the sorbent types
        return render_template('index.html', sorbent_types=sorbent_types)
    except Exception as e:
        logger.error(f"Error rendering app page: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error loading app page", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        logger.debug(f"Received input data: {input_data}")
        
        # Validate input data
        required_fields = [
            'Temperature (°C)', 'Pressure (bar)', 'Flow Rate (m³/h)',
            'CO2 Concentration (%)', 'CO2 Outlet Concentration (%)',
            'Capture Efficiency (%)', 'Column Height (m)', 'Cycle Time (min)',
            'Energy Consumption (MJ/kg CO2)', 'Pore Size (nm)',
            'Reactor Volume (m³)', 'Reboiler Duty (kW)',
            'Regeneration Temp (°C)', 'Residence Time (s)',
            'Solvent Concentration (%)', 'Stability (1-10)',
            'Surface Area (m²/g)', 'pH Level', 'Sorbent Type'
        ]
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            logger.error(error_msg)
            return jsonify({
                'error': error_msg,
                'missing_fields': missing_fields
            }), 400

        # Convert string values to float where appropriate
        for field in input_data:
            if field != 'Sorbent Type' and isinstance(input_data[field], str):
                try:
                    input_data[field] = float(input_data[field])
                except ValueError:
                    error_msg = f"Invalid value for {field}. Expected a number, got {input_data[field]}"
                    logger.error(error_msg)
                    return jsonify({
                        'error': error_msg,
                        'field': field,
                        'value': input_data[field]
                    }), 400

        # Make prediction
        logger.info("Making prediction with input data...")
        prediction = model.predict(input_data)
        logger.info(f"Prediction result: {prediction}")
        
        logger.info("Analyzing parameter sensitivity...")
        recommendations = model.analyze_parameter_sensitivity(input_data)
        logger.info(f"Recommendations generated: {recommendations}")
        
        # Store results in session
        session['prediction_results'] = {
            'prediction': prediction,
            'recommendations': recommendations,
            'input_data': input_data
        }
        
        return jsonify({'redirect': url_for('results_page')})
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            'error': error_msg,
            'details': traceback.format_exc()
        }), 500

@app.route('/results')
def results_page():
    # Retrieve results from session
    results = session.get('prediction_results')
    
    if not results:
        # If no results in session, redirect back to the app page
        return redirect(url_for('app_page'))
        
    # Clear the session data after retrieving it
    session.pop('prediction_results', None)
    
    # Render the results template with the data
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, port=1000)
