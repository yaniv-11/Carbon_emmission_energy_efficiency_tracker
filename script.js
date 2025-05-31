document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form values
    const inputData = {
        'Temperature (°C)': parseFloat(document.getElementById('temperature').value),
        'Pressure (bar)': parseFloat(document.getElementById('pressure').value),
        'CO2 Concentration (%)': parseFloat(document.getElementById('co2Concentration').value),
        'Flow Rate (m³/h)': parseFloat(document.getElementById('flowRate').value),
        'Solvent Concentration (%)': parseFloat(document.getElementById('solventConcentration').value),
        'Sorbent Type': document.getElementById('sorbentType').value,
        'Capture Efficiency (%)': parseFloat(document.getElementById('captureEfficiency').value),
        'Energy Consumption (MJ/kg CO2)': parseFloat(document.getElementById('energyConsumption').value),
        // Add default values for other parameters
        'CO2 Outlet Concentration (%)': 2,
        'Column Height (m)': 10,
        'Cycle Time (min)': 30,
        'Pore Size (nm)': 50,
        'Reactor Volume (m³)': 5,
        'Reboiler Duty (kW)': 100,
        'Regeneration Temp (°C)': 120,
        'Residence Time (s)': 60,
        'Stability (1-10)': 8,
        'Surface Area (m²/g)': 200,
        'pH Level': 7
    };
    
    try {
        // Show loading state
        const submitButton = this.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        submitButton.innerHTML = 'Loading...';
        
        // Make API request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(inputData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Display predictions
            document.getElementById('predictions').classList.remove('d-none');
            document.getElementById('co2Prediction').textContent = 
                `Predicted CO2 Emissions: ${data.predictions['CO2 Emitted (kg/hr)'].toFixed(2)} kg/hr`;
            document.getElementById('energyPrediction').textContent = 
                `Predicted Energy Efficiency: ${data.predictions['Energy Used Efficiently (MJ)'].toFixed(2)} MJ`;
            
            // Display recommendations
            document.getElementById('recommendations').classList.remove('d-none');
            
            // Clear previous recommendations
            document.getElementById('increaseList').innerHTML = '';
            document.getElementById('decreaseList').innerHTML = '';
            
            // Add increase recommendations
            if (data.recommendations.increase.length > 0) {
                data.recommendations.increase.forEach(rec => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.innerHTML = `
                        <strong>${rec.parameter}</strong><br>
                        Current: ${rec.current_value}<br>
                        CO2 Improvement: ${rec.co2_improvement.toFixed(2)} kg/hr<br>
                        Energy Improvement: ${rec.energy_improvement.toFixed(2)} MJ
                    `;
                    document.getElementById('increaseList').appendChild(li);
                });
            } else {
                document.getElementById('increaseRecommendations').classList.add('d-none');
            }
            
            // Add decrease recommendations
            if (data.recommendations.decrease.length > 0) {
                data.recommendations.decrease.forEach(rec => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.innerHTML = `
                        <strong>${rec.parameter}</strong><br>
                        Current: ${rec.current_value}<br>
                        CO2 Improvement: ${rec.co2_improvement.toFixed(2)} kg/hr<br>
                        Energy Improvement: ${rec.energy_improvement.toFixed(2)} MJ
                    `;
                    document.getElementById('decreaseList').appendChild(li);
                });
            } else {
                document.getElementById('decreaseRecommendations').classList.add('d-none');
            }
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        // Reset button state
        submitButton.disabled = false;
        submitButton.innerHTML = 'Get Predictions';
    }
}); 