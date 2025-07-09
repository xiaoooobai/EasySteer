// Page switching functionality
function switchPage(pageType) {
    // Hide all pages
    document.querySelectorAll('.page-content').forEach(page => {
        page.classList.remove('active');
    });
    
    // Remove active class from all navigation buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show the selected page
    if (pageType === 'inference') {
        document.getElementById('inference-page').classList.add('active');
        document.querySelector('[onclick="switchPage(\'inference\')"]').classList.add('active');
        loadConfigOptions();
    } else if (pageType === 'training') {
        document.getElementById('training-page').classList.add('active');
        document.querySelector('[onclick="switchPage(\'training\')"]').classList.add('active');
        loadTrainConfigOptions();
    } else if (pageType === 'extraction') {
        document.getElementById('extraction-page').classList.add('active');
        document.querySelector('[onclick="switchPage(\'extraction\')"]').classList.add('active');
        loadExtractConfigOptions();
    }
}

// Initialize synchronization for sliders and number inputs
const scaleSlider = document.getElementById('scaleSlider');
const scaleInput = document.getElementById('scale');
const temperatureSlider = document.getElementById('temperatureSlider');
const temperatureInput = document.getElementById('temperature');
const repetitionPenaltySlider = document.getElementById('repetitionPenaltySlider');
const repetitionPenaltyInput = document.getElementById('repetitionPenalty');

// Scale slider synchronization
scaleSlider.addEventListener('input', function() {
    scaleInput.value = this.value;
});

scaleInput.addEventListener('input', function() {
    scaleSlider.value = this.value;
});

// Temperature slider synchronization
temperatureSlider.addEventListener('input', function() {
    temperatureInput.value = this.value;
});

temperatureInput.addEventListener('input', function() {
    temperatureSlider.value = this.value;
});

// Repetition Penalty slider synchronization
repetitionPenaltySlider.addEventListener('input', function() {
    repetitionPenaltyInput.value = this.value;
});

repetitionPenaltyInput.addEventListener('input', function() {
    repetitionPenaltySlider.value = this.value;
});

// Handle file selection
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('localPath').value = file.name;
    }
});

// Parse list input
function parseListInput(input) {
    if (!input || input.trim() === '') {
        return null;
    }
    return input.split(',').map(item => parseInt(item.trim())).filter(num => !isNaN(num));
}

// Submit configuration
async function submitConfiguration() {
    const config = {
        // Model configuration
        model_path: document.getElementById('modelPath').value,
        gpu_devices: document.getElementById('gpuDevices').value,
        instruction: document.getElementById('instruction').value,
        
        // Sampling parameters
        sampling_params: {
            temperature: parseFloat(document.getElementById('temperature').value),
            max_tokens: parseInt(document.getElementById('maxTokens').value),
            repetition_penalty: parseFloat(document.getElementById('repetitionPenalty').value)
        },
        
        // Steer Vector configuration
        steer_vector_name: document.getElementById('steerVectorName').value,
        steer_vector_id: parseInt(document.getElementById('steerVectorId').value),
        steer_vector_local_path: document.getElementById('localPath').value,
        scale: parseFloat(document.getElementById('scale').value),
        algorithm: document.getElementById('algorithm').value,
        target_layers: parseListInput(document.getElementById('targetLayers').value),
        prefill_trigger_tokens: parseListInput(document.getElementById('prefillTriggerTokens').value),
        prefill_trigger_positions: parseListInput(document.getElementById('prefillTriggerPositions').value),
        generate_trigger_tokens: parseListInput(document.getElementById('generateTriggerTokens').value),
        debug: document.getElementById('debug').checked
    };

    // Validate required fields
    if (!config.steer_vector_name || !config.steer_vector_id || !config.steer_vector_local_path || !config.model_path || !config.instruction) {
        showError(t('required_fields_error'));
        return;
    }

    // Show loading state
    const submitButton = document.querySelector('.btn-primary');
    const originalHTML = submitButton.innerHTML;
    submitButton.innerHTML = '<span class="loading"></span> ' + t('generating');
    submitButton.disabled = true;

    try {
        // Send request to the backend - auto-detect current host
        const apiUrl = `${window.location.protocol}//${window.location.hostname}:5000/api/generate`;
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept-Language': currentLanguage
            },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (response.ok) {
            showResponse(data);
        } else {
            showError(data.error || t('submit_failed'));
        }
    } catch (error) {
        console.error('Error:', error);
        showError(t('network_error'));
    } finally {
        // Restore button state
        submitButton.innerHTML = originalHTML;
        submitButton.disabled = false;
    }
}

// Display response
function showResponse(data) {
    const responseDiv = document.getElementById('response');
    const responseContent = document.getElementById('responseContent');
    const errorDiv = document.getElementById('error');
    
    // Format and display the generation result
    if (data.generated_text) {
        responseContent.textContent = data.generated_text;
    } else {
        responseContent.textContent = JSON.stringify(data, null, 2);
    }
    
    responseDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    
    // Scroll to the result
    responseDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display error message
function showError(message) {
    const errorDiv = document.getElementById('error');
    const errorContent = document.getElementById('errorContent');
    const responseDiv = document.getElementById('response');
    
    errorContent.textContent = message;
    errorDiv.style.display = 'block';
    responseDiv.style.display = 'none';
    
    // Scroll to the error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Reset form
function resetForm() {
    // Model configuration
    document.getElementById('modelPath').value = '';
    document.getElementById('gpuDevices').value = '0';
    document.getElementById('instruction').value = '';
    
    // Sampling parameters
    document.getElementById('temperature').value = '0.0';
    document.getElementById('temperatureSlider').value = '0.0';
    document.getElementById('maxTokens').value = '128';
    document.getElementById('repetitionPenalty').value = '1.1';
    document.getElementById('repetitionPenaltySlider').value = '1.1';
    
    // Steer Vector configuration
    document.getElementById('steerVectorName').value = '';
    document.getElementById('steerVectorId').value = '';
    document.getElementById('localPath').value = '';
    document.getElementById('scale').value = '1.0';
    document.getElementById('scaleSlider').value = '1.0';
    document.getElementById('algorithm').value = 'direct';
    document.getElementById('targetLayers').value = '';
    document.getElementById('prefillTriggerTokens').value = '';
    document.getElementById('prefillTriggerPositions').value = '';
    document.getElementById('generateTriggerTokens').value = '';
    document.getElementById('debug').checked = false;
    document.getElementById('fileInput').value = '';
    
    // Hide results and errors
    document.getElementById('response').style.display = 'none';
    document.getElementById('error').style.display = 'none';
}

// Import selected configuration
async function importSelectedConfig() {
    const configSelect = document.getElementById('configSelect');
    const selectedConfig = configSelect.value;
    
    if (!selectedConfig) {
        showError(t('error_select_config'));
        return;
    }
    
    try {
        showStatus(t('importing_config', { configName: configSelect.options[configSelect.selectedIndex].text }), 'info');
        
        // Get config file from the backend
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/config/${selectedConfig}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const config = await response.json();
        
        // Set model configuration
        document.getElementById('modelPath').value = config.model.path || '';
        document.getElementById('gpuDevices').value = config.model.gpu_devices || '';
        document.getElementById('instruction').value = config.model.instruction || '';
        
        // Set sampling parameters
        document.getElementById('temperature').value = config.sampling.temperature || '0.0';
        document.getElementById('temperatureSlider').value = config.sampling.temperature || '0.0';
        document.getElementById('maxTokens').value = config.sampling.max_tokens || '128';
        document.getElementById('repetitionPenalty').value = config.sampling.repetition_penalty || '1.1';
        document.getElementById('repetitionPenaltySlider').value = config.sampling.repetition_penalty || '1.1';
        
        // Set Steer Vector configuration
        document.getElementById('steerVectorName').value = config.steer_vector.name || '';
        document.getElementById('steerVectorId').value = config.steer_vector.id || '';
        document.getElementById('localPath').value = config.steer_vector.path || '';
        document.getElementById('scale').value = config.steer_vector.scale || '1.0';
        document.getElementById('scaleSlider').value = config.steer_vector.scale || '1.0';
        document.getElementById('algorithm').value = config.steer_vector.algorithm || 'direct';
        document.getElementById('targetLayers').value = config.steer_vector.target_layers || '';
        
        // Set trigger configuration - based on algorithm type
        if (config.steer_vector.algorithm === 'loreft') {
            // LoReft algorithm uses position triggers
            document.getElementById('prefillTriggerPositions').value = config.steer_vector.prefill_positions || '';
            document.getElementById('prefillTriggerTokens').value = '';
            document.getElementById('generateTriggerTokens').value = '';
        } else {
            // Direct algorithm uses token triggers
            document.getElementById('prefillTriggerTokens').value = config.steer_vector.prefill_trigger_tokens || '';
            document.getElementById('generateTriggerTokens').value = config.steer_vector.generate_trigger_tokens || '';
            document.getElementById('prefillTriggerPositions').value = '';
        }
        
        document.getElementById('debug').checked = config.steer_vector.debug || false;
        
        showResponse({
            message: t('import_success_message'),
            description: t('import_success_description', { configName: configSelect.options[configSelect.selectedIndex].text })
        });
        
        // Clear selection box
        configSelect.value = '';
        
    } catch (error) {
        showError(t('import_fail_error') + ': ' + error.message);
    }
}

// Restart backend
async function restartBackend() {
    const isConfirmed = confirm(t('confirm_restart'));
    if (!isConfirmed) {
        return;
    }
    
    try {
        showStatus(t('restarting_backend'), 'info');
        
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/restart`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showResponse({
                message: t('restart_success_message'),
                description: t('restart_success_description')
            });
        } else {
            throw new Error(result.message || t('restart_fail_error'));
        }
        
    } catch (error) {
        showError(t('restart_fail_error') + ': ' + error.message);
    }
}

// Display status message
function showStatus(message, type = 'info') {
    // Can use showResponse or showError directly
    if (type === 'error') {
        showError(message);
    } else {
        showResponse({ message: message });
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        submitConfiguration();
    }
    // Ctrl/Cmd + R to reset (prevent default refresh)
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        resetForm();
    }
});

// Dynamically load configuration options
async function loadConfigOptions() {
    try {
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/configs`);
        if (response.ok) {
            const data = await response.json();
            const configSelect = document.getElementById('configSelect');
            
            // Clear existing options (except the default one)
            while (configSelect.children.length > 1) {
                configSelect.removeChild(configSelect.lastChild);
            }
            
            // Add dynamically loaded configuration options
            data.configs.forEach(config => {
                const option = document.createElement('option');
                option.value = config.name;
                option.textContent = config.display_name;
                configSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.log('Could not load configuration options, using default options');
    }
}

// Add tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Load configuration options
    loadConfigOptions();
    loadTrainConfigOptions();
    
    // Add focus effect for all input fields with help-text
    const inputs = document.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            const helpText = this.parentElement.querySelector('.help-text') || 
                           this.parentElement.parentElement.querySelector('.help-text');
            if (helpText) {
                helpText.style.color = 'var(--primary-color)';
            }
        });
        
        input.addEventListener('blur', function() {
            const helpText = this.parentElement.querySelector('.help-text') || 
                           this.parentElement.parentElement.querySelector('.help-text');
            if (helpText) {
                helpText.style.color = 'var(--text-secondary)';
            }
        });
    });
});

// Training related functions

// Start training
async function startTraining() {
    const trainConfig = {
        // Model configuration
        model_path: document.getElementById('trainModelPath').value,
        gpu_devices: document.getElementById('trainGpuDevices').value,
        output_dir: document.getElementById('outputDir').value,
        
        // ReFT configuration
        reft_config: {
            layer: parseInt(document.getElementById('trainLayer').value),
            component: document.getElementById('trainComponent').value,
            low_rank_dimension: parseInt(document.getElementById('lowRankDim').value)
        },
        
        // Training parameters
        training_args: {
            num_train_epochs: parseFloat(document.getElementById('numEpochs').value),
            per_device_train_batch_size: parseInt(document.getElementById('batchSize').value),
            learning_rate: parseFloat(document.getElementById('learningRate').value),
            logging_steps: parseInt(document.getElementById('loggingSteps').value),
            save_steps: parseInt(document.getElementById('saveSteps').value)
        },
        
        // Training data
        training_examples: document.getElementById('trainingExamples').value
    };

    // Validate required fields
    if (!trainConfig.model_path || !trainConfig.output_dir || !trainConfig.training_examples) {
        showTrainError(t('required_fields_error'));
        return;
    }

    // Validate training data format
    try {
        const examples = JSON.parse(trainConfig.training_examples);
        if (!Array.isArray(examples) || examples.length === 0) {
            throw new Error('Training examples must be a non-empty array');
        }
        // Validate format of each example
        examples.forEach((example, index) => {
            if (!Array.isArray(example) || example.length !== 2) {
                throw new Error(`Example ${index} must be an array with exactly 2 elements [input, output]`);
            }
        });
        trainConfig.training_examples = examples;
    } catch (error) {
        showTrainError(t('train_data_format_error') + ': ' + error.message);
        return;
    }

    // Show loading state
    const trainButton = document.querySelector('[onclick="startTraining()"]');
    const originalHTML = trainButton.innerHTML;
    trainButton.innerHTML = '<span class="loading"></span> ' + t('training_in_progress');
    trainButton.disabled = true;

    try {
        // Send request to the backend
        const apiUrl = `${window.location.protocol}//${window.location.hostname}:5000/api/train`;
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept-Language': currentLanguage
            },
            body: JSON.stringify(trainConfig)
        });

        const data = await response.json();

        if (response.ok) {
            showTrainResponse(data);
        } else {
            showTrainError(data.error || t('training_failed_error'));
        }
    } catch (error) {
        console.error('Training error:', error);
        showTrainError(t('network_error'));
    } finally {
        // Restore button state
        trainButton.innerHTML = originalHTML;
        trainButton.disabled = false;
    }
}

// Training status polling interval ID
let trainingStatusInterval = null;

// Display training response
function showTrainResponse(data) {
    const responseDiv = document.getElementById('trainResponse');
    const responseContent = document.getElementById('trainResponseContent');
    const errorDiv = document.getElementById('trainError');
    
    if (data.success) {
        // If training started successfully, display progress UI and start polling
        responseContent.innerHTML = createTrainingProgressHTML();
        startTrainingStatusPolling();
    } else {
        // Display error message
        responseContent.textContent = JSON.stringify(data, null, 2);
    }
    
    responseDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    
    // Scroll to the result
    responseDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Create training progress HTML
function createTrainingProgressHTML() {
    return `
        <div class="logs-container" id="trainingLogs">
            <div class="log-entry status-entry">${t('initializing_training')}</div>
        </div>
    `;
}

// Start polling for training status
function startTrainingStatusPolling() {
    // Clear previous polling
    if (trainingStatusInterval) {
        clearInterval(trainingStatusInterval);
    }
    
    // Get status immediately once
    updateTrainingStatus();
    
    // Poll every 2 seconds
    trainingStatusInterval = setInterval(updateTrainingStatus, 2000);
}

// Stop polling for training status
function stopTrainingStatusPolling() {
    if (trainingStatusInterval) {
        clearInterval(trainingStatusInterval);
        trainingStatusInterval = null;
    }
}

// Update training status
async function updateTrainingStatus() {
    try {
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/train-status`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const status = await response.json();
        displayTrainingStatus(status);
        
        // If training is finished, stop polling
        if (!status.is_training) {
            stopTrainingStatusPolling();
        }
        
    } catch (error) {
        console.error('Failed to get training status:', error);
        // Stop polling on network error
        stopTrainingStatusPolling();
    }
}

// Display training status
function displayTrainingStatus(status) {
    const logsContainer = document.getElementById('trainingLogs');
    if (!logsContainer) return;
    
    // Create log content array
    let allLogs = [];
    
    // Add status message as a special log entry
    if (status.status_message) {
        let statusClass = 'status-entry';
        
        if (status.error_message) {
            statusClass += ' error-status';
        } else if (!status.is_training && status.status_message.includes('完成')) {
            statusClass += ' success-status';
        } else if (status.is_training) {
            statusClass += ' training-status';
        }
        
        allLogs.push(`<div class="log-entry ${statusClass}">
            <strong>📊 ${status.status_message}</strong>
        </div>`);
    }
    
    // Add normal training logs
    if (status.logs && status.logs.length > 0) {
        const formattedLogs = status.logs.slice(-20).map(log => {
            // Add colors for different types of information
            let coloredLog = log;
            
            // Timestamp
            coloredLog = coloredLog.replace(/\[\d{2}:\d{2}:\d{2}\]/g, '<span style="color: #6b7280;">$&</span>');
            
            // Epoch info
            coloredLog = coloredLog.replace(/Epoch: [\d.]+/g, '<span style="color: #2563eb; font-weight: 600;">$&</span>');
            
            // Loss info
            coloredLog = coloredLog.replace(/Loss: [\d.]+/g, '<span style="color: #dc2626; font-weight: 600;">$&</span>');
            
            // Gradient info
            coloredLog = coloredLog.replace(/Grad: [\d.]+/g, '<span style="color: #ea580c; font-weight: 600;">$&</span>');
            
            // Learning rate
            coloredLog = coloredLog.replace(/LR: [\d.e-]+/g, '<span style="color: #16a34a; font-weight: 600;">$&</span>');
            
            // Runtime and speed
            coloredLog = coloredLog.replace(/Runtime: [\d.]+s/g, '<span style="color: #7c3aed; font-weight: 500;">$&</span>');
            coloredLog = coloredLog.replace(/Speed: [\d.]+ samples\/s/g, '<span style="color: #c2410c; font-weight: 500;">$&</span>');
            
            return `<div class="log-entry">${coloredLog}</div>`;
        });
        
        allLogs = allLogs.concat(formattedLogs);
    }
    
    // If there are no logs, show waiting message
    if (allLogs.length === 0) {
        allLogs.push(`<div class="log-entry status-entry">${t('waiting_for_training')}</div>`);
    }
    
    // Update log container
    logsContainer.innerHTML = allLogs.join('');
    
    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Display training error message
function showTrainError(message) {
    const errorDiv = document.getElementById('trainError');
    const errorContent = document.getElementById('trainErrorContent');
    const responseDiv = document.getElementById('trainResponse');
    
    errorContent.textContent = message;
    errorDiv.style.display = 'block';
    responseDiv.style.display = 'none';
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Reset training form
function resetTrainForm() {
    // Model configuration
    document.getElementById('trainModelPath').value = '';
    document.getElementById('trainGpuDevices').value = '0';
    document.getElementById('outputDir').value = './results/loreft';
    
    // ReFT configuration
    document.getElementById('trainLayer').value = '8';
    document.getElementById('trainComponent').value = 'block_output';
    document.getElementById('lowRankDim').value = '4';
    
    // Training parameters
    document.getElementById('numEpochs').value = '100';
    document.getElementById('batchSize').value = '10';
    document.getElementById('learningRate').value = '0.004';
    document.getElementById('loggingSteps').value = '40';
    document.getElementById('saveSteps').value = '500';
    
    // Training data
    document.getElementById('trainingExamples').value = '';
    
    // Stop training status polling
    stopTrainingStatusPolling();
    
    // Hide results and errors
    document.getElementById('trainResponse').style.display = 'none';
    document.getElementById('trainError').style.display = 'none';
}

// Dynamically load training configuration options
async function loadTrainConfigOptions() {
    try {
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/train-configs`);
        if (response.ok) {
            const data = await response.json();
            const trainConfigSelect = document.getElementById('trainConfigSelect');
            
            // Clear existing options (except the default one)
            while (trainConfigSelect.children.length > 1) {
                trainConfigSelect.removeChild(trainConfigSelect.lastChild);
            }
            
            // Add dynamically loaded configuration options
            data.configs.forEach(config => {
                const option = document.createElement('option');
                option.value = config.name;
                option.textContent = config.display_name;
                trainConfigSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.log('Could not load training configuration options');
    }
}

// Import selected training configuration
async function importSelectedTrainConfig() {
    const trainConfigSelect = document.getElementById('trainConfigSelect');
    const selectedConfig = trainConfigSelect.value;
    
    if (!selectedConfig) {
        showTrainError(t('error_select_train_config'));
        return;
    }
    
    try {
        showTrainResponse({ message: t('importing_train_config', { configName: trainConfigSelect.options[trainConfigSelect.selectedIndex].text }) });
        
        // Get training config file from the backend
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/train-config/${selectedConfig}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const config = await response.json();
        
        // Set model configuration
        document.getElementById('trainModelPath').value = config.model.path || '';
        document.getElementById('trainGpuDevices').value = config.model.gpu_devices || '0';
        document.getElementById('outputDir').value = config.training.output_dir || './results/loreft';
        
        // Set ReFT configuration
        document.getElementById('trainLayer').value = config.reft.layer || '8';
        document.getElementById('trainComponent').value = config.reft.component || 'block_output';
        document.getElementById('lowRankDim').value = config.reft.low_rank_dimension || '4';
        
        // Set training parameters
        document.getElementById('numEpochs').value = config.training.num_train_epochs || '100';
        document.getElementById('batchSize').value = config.training.per_device_train_batch_size || '10';
        document.getElementById('learningRate').value = config.training.learning_rate || '0.004';
        document.getElementById('loggingSteps').value = config.training.logging_steps || '40';
        document.getElementById('saveSteps').value = config.training.save_steps || '500';
        
        // Set training data
        if (config.data && config.data.training_examples) {
            document.getElementById('trainingExamples').value = JSON.stringify(config.data.training_examples, null, 2);
        }
        
        showTrainResponse({
            message: t('train_import_success_message'),
            description: t('train_import_success_description', { configName: trainConfigSelect.options[trainConfigSelect.selectedIndex].text })
        });
        
        // Clear selection box
        trainConfigSelect.value = '';
        
    } catch (error) {
        showTrainError(t('train_import_fail_error') + ': ' + error.message);
    }
}

// Dynamically load extraction configuration options
async function loadExtractConfigOptions() {
    try {
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/extract-configs`);
        if (response.ok) {
            const data = await response.json();
            const extractConfigSelect = document.getElementById('extractConfigSelect');
            
            // Clear existing options (except the default one)
            while (extractConfigSelect.children.length > 1) {
                extractConfigSelect.removeChild(extractConfigSelect.lastChild);
            }
            
            // Add dynamically loaded configuration options
            data.configs.forEach(config => {
                const option = document.createElement('option');
                option.value = config.name;
                option.textContent = config.display_name;
                extractConfigSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.log('Could not load extraction configuration options, using default options');
    }
}

// Import selected extraction configuration
async function importSelectedExtractConfig() {
    const extractConfigSelect = document.getElementById('extractConfigSelect');
    const selectedConfig = extractConfigSelect.value;
    
    if (!selectedConfig) {
        showExtractError(t('error_select_extract_config'));
        return;
    }
    
    try {
        showExtractResponse({ message: t('importing_extract_config', { configName: extractConfigSelect.options[extractConfigSelect.selectedIndex].text }) });
        
        // Get extraction config file from the backend
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/extract-config/${selectedConfig}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const config = await response.json();
        
        // Set model configuration
        document.getElementById('extractModelPath').value = config.model_path || '';
        document.getElementById('extractGpuDevices').value = config.gpu_devices || '0';
        
        // Set extraction method configuration
        document.getElementById('extractMethod').value = config.method || 'diffmean';
        document.getElementById('extractTargetLayer').value = config.target_layer || '';
        document.getElementById('extractTokenPos').value = config.token_pos || '-1';
        document.getElementById('extractNormalize').checked = config.normalize !== false;
        
        // Set SAE-specific parameters
        if (config.method === 'sae') {
            document.getElementById('saeParamsPath').value = config.sae_params_path || '';
            document.getElementById('saeCombinationMode').value = config.combination_mode || 'weighted_all';
            document.getElementById('saeTopK').value = config.top_k || 100;
        }
        
        // Set sample data
        if (config.positive_samples && Array.isArray(config.positive_samples)) {
            document.getElementById('extractPositiveSamples').value = config.positive_samples.join('\n');
        }
        if (config.negative_samples && Array.isArray(config.negative_samples)) {
            document.getElementById('extractNegativeSamples').value = config.negative_samples.join('\n');
        }
        
        // Set output configuration
        document.getElementById('extractOutputPath').value = config.output_path || './extracted_vectors/control_vector.safetensors';
        document.getElementById('extractVectorName').value = config.vector_name || 'extracted_vector';
        
        // Trigger method option update
        updateExtractionMethodOptions();
        
        showExtractResponse({
            message: t('extract_import_success_message'),
            description: t('extract_import_success_description', { configName: extractConfigSelect.options[extractConfigSelect.selectedIndex].text })
        });
        
        // Clear selection box
        extractConfigSelect.value = '';
        
    } catch (error) {
        showExtractError(t('extract_import_fail_error') + ': ' + error.message);
    }
}

// Extraction related functions

// Update extraction method options
function updateExtractionMethodOptions() {
    const method = document.getElementById('extractMethod').value;
    const saeOptions = document.getElementById('saeOptions');
    const saeCombinationMode = document.getElementById('saeCombinationMode');
    const saeTopKGroup = document.getElementById('saeTopKGroup');
    
    // Hide all method-specific options
    saeOptions.style.display = 'none';
    
    // Show options based on the selected method
    if (method === 'sae') {
        saeOptions.style.display = 'block';
        // Show/hide top-k option based on combination mode
        if (saeCombinationMode.value === 'weighted_top_k') {
            saeTopKGroup.style.display = 'block';
        } else {
            saeTopKGroup.style.display = 'none';
        }
    }
}

// Handle SAE combination mode change
document.addEventListener('DOMContentLoaded', function() {
    const saeCombinationMode = document.getElementById('saeCombinationMode');
    if (saeCombinationMode) {
        saeCombinationMode.addEventListener('change', function() {
            const saeTopKGroup = document.getElementById('saeTopKGroup');
            if (this.value === 'weighted_top_k') {
                saeTopKGroup.style.display = 'block';
            } else {
                saeTopKGroup.style.display = 'none';
            }
        });
    }
});

// Extraction status polling interval ID
let extractionStatusInterval = null;

// Start extraction
async function startExtraction() {
    // Get form data
    const extractConfig = {
        // Model configuration
        model_path: document.getElementById('extractModelPath').value,
        gpu_devices: document.getElementById('extractGpuDevices').value,
        
        // Extraction method configuration
        method: document.getElementById('extractMethod').value,
        target_layer: document.getElementById('extractTargetLayer').value ? 
            parseInt(document.getElementById('extractTargetLayer').value) : null,
        token_pos: document.getElementById('extractTokenPos').value,
        normalize: document.getElementById('extractNormalize').checked,
        
        // Sample data
        positive_samples: document.getElementById('extractPositiveSamples').value.trim().split('\n').filter(s => s.trim()),
        negative_samples: document.getElementById('extractNegativeSamples').value.trim().split('\n').filter(s => s.trim()),
        
        // Output configuration
        output_path: document.getElementById('extractOutputPath').value,
        vector_name: document.getElementById('extractVectorName').value
    };
    
    // SAE-specific configuration
    if (extractConfig.method === 'sae') {
        extractConfig.sae_params_path = document.getElementById('saeParamsPath').value;
        extractConfig.combination_mode = document.getElementById('saeCombinationMode').value;
        if (extractConfig.combination_mode === 'weighted_top_k') {
            extractConfig.top_k = parseInt(document.getElementById('saeTopK').value);
        }
    }
    
    // Validate required fields
    if (!extractConfig.model_path || !extractConfig.output_path || 
        extractConfig.positive_samples.length === 0 || extractConfig.negative_samples.length === 0) {
        showExtractError(t('required_fields_error'));
        return;
    }
    
    // If SAE method, validate params path
    if (extractConfig.method === 'sae' && !extractConfig.sae_params_path) {
        showExtractError(t('sae_path_error'));
        return;
    }
    
    // Show loading state
    const extractButton = document.querySelector('[onclick="startExtraction()"]');
    const originalHTML = extractButton.innerHTML;
    extractButton.innerHTML = '<span class="loading"></span> ' + t('extracting_in_progress');
    extractButton.disabled = true;
    
    try {
        // Send request to the backend
        const apiUrl = `${window.location.protocol}//${window.location.hostname}:5000/api/extract`;
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept-Language': currentLanguage
            },
            body: JSON.stringify(extractConfig)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showExtractResponse(data);
        } else {
            showExtractError(data.error || t('extraction_failed_error'));
        }
    } catch (error) {
        console.error('Extraction error:', error);
        showExtractError(t('network_error'));
    } finally {
        // Restore button state
        extractButton.innerHTML = originalHTML;
        extractButton.disabled = false;
    }
}

// Display extraction response
function showExtractResponse(data) {
    const responseDiv = document.getElementById('extractResponse');
    const responseContent = document.getElementById('extractResponseContent');
    const errorDiv = document.getElementById('extractError');
    
    if (data.success) {
        // Create extraction progress HTML
        responseContent.innerHTML = createExtractionProgressHTML();
        startExtractionStatusPolling();
    } else {
        // Display detailed results
        let resultHTML = '<div class="extraction-result">';
        resultHTML += '<div class="result-info">';
        resultHTML += `<p><strong>${t('status_label')}:</strong> ${data.message || t('extraction_complete')}</p>`;
        if (data.output_path) {
            resultHTML += `<p><strong>${t('output_file_label')}:</strong> <code>${data.output_path}</code></p>`;
        }
        if (data.metadata) {
            resultHTML += `<p><strong>${t('metadata_label')}:</strong></p>`;
            resultHTML += '<pre>' + JSON.stringify(data.metadata, null, 2) + '</pre>';
        }
        resultHTML += '</div>';
        resultHTML += '</div>';
        
        responseContent.innerHTML = resultHTML;
    }
    
    responseDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    
    // Scroll to the result
    responseDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Create extraction progress HTML
function createExtractionProgressHTML() {
    return `
        <div class="logs-container" id="extractionLogs">
            <div class="log-entry status-entry">${t('initializing_extraction')}</div>
        </div>
    `;
}

// Start polling for extraction status
function startExtractionStatusPolling() {
    // Clear previous polling
    if (extractionStatusInterval) {
        clearInterval(extractionStatusInterval);
    }
    
    // Get status immediately once
    updateExtractionStatus();
    
    // Poll every 2 seconds
    extractionStatusInterval = setInterval(updateExtractionStatus, 2000);
}

// Stop polling for extraction status
function stopExtractionStatusPolling() {
    if (extractionStatusInterval) {
        clearInterval(extractionStatusInterval);
        extractionStatusInterval = null;
    }
}

// Update extraction status
async function updateExtractionStatus() {
    try {
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:5000/api/extract-status`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const status = await response.json();
        displayExtractionStatus(status);
        
        // If extraction is finished, stop polling
        if (!status.is_extracting) {
            stopExtractionStatusPolling();
        }
        
    } catch (error) {
        console.error('Failed to get extraction status:', error);
        // Stop polling on network error
        stopExtractionStatusPolling();
    }
}

// Display extraction status
function displayExtractionStatus(status) {
    const logsContainer = document.getElementById('extractionLogs');
    if (!logsContainer) return;
    
    // Create log content array
    let allLogs = [];
    
    // Add status message
    if (status.status_message) {
        let statusClass = 'status-entry';
        
        if (status.error_message) {
            statusClass += ' error-status';
        } else if (!status.is_extracting && status.status_message.includes('完成')) {
            statusClass += ' success-status';
        } else if (status.is_extracting) {
            statusClass += ' training-status';
        }
        
        allLogs.push(`<div class="log-entry ${statusClass}">
            <strong>📊 ${status.status_message}</strong>
        </div>`);
    }
    
    // Add extraction logs
    if (status.logs && status.logs.length > 0) {
        const formattedLogs = status.logs.slice(-20).map(log => {
            return `<div class="log-entry">${log}</div>`;
        });
        
        allLogs = allLogs.concat(formattedLogs);
    }
    
    // Add result info
    if (status.result) {
        allLogs.push(`<div class="log-entry status-entry success-status"><strong>✅ ${t('extraction_complete')}</strong></div>`);
        if (status.result.output_path) {
            allLogs.push(`<div class="log-entry">📁 ${t('output_file_label')}: <code>${status.result.output_path}</code></div>`);
        }
        if (status.result.layers_extracted) {
            allLogs.push(`<div class="log-entry">📊 ${t('layers_extracted_label')}: ${status.result.layers_extracted}</div>`);
        }
    }
    
    // If there are no logs, show waiting message
    if (allLogs.length === 0) {
        allLogs.push(`<div class="log-entry status-entry">${t('waiting_for_extraction')}</div>`);
    }
    
    // Update log container
    logsContainer.innerHTML = allLogs.join('');
    
    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Display extraction error message
function showExtractError(message) {
    const errorDiv = document.getElementById('extractError');
    const errorContent = document.getElementById('extractErrorContent');
    const responseDiv = document.getElementById('extractResponse');
    
    errorContent.textContent = message;
    errorDiv.style.display = 'block';
    responseDiv.style.display = 'none';
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Reset extraction form
function resetExtractForm() {
    // Model configuration
    document.getElementById('extractModelPath').value = '';
    document.getElementById('extractGpuDevices').value = '0';
    
    // Extraction method configuration
    document.getElementById('extractMethod').value = 'lat';
    document.getElementById('extractTargetLayer').value = '';
    document.getElementById('extractTokenPos').value = '-1';
    document.getElementById('extractNormalize').checked = true;
    
    // SAE-specific configuration
    document.getElementById('saeParamsPath').value = '';
    document.getElementById('saeCombinationMode').value = 'weighted_all';
    document.getElementById('saeTopK').value = '100';
    
    // Sample data
    document.getElementById('extractPositiveSamples').value = '';
    document.getElementById('extractNegativeSamples').value = '';
    
    // Output configuration
    document.getElementById('extractOutputPath').value = './extracted_vectors/control_vector.safetensors';
    document.getElementById('extractVectorName').value = 'extracted_vector';
    
    // Stop extraction status polling
    stopExtractionStatusPolling();
    
    // Hide results and errors
    document.getElementById('extractResponse').style.display = 'none';
    document.getElementById('extractError').style.display = 'none';
    
    // Update method options display
    updateExtractionMethodOptions();
} 