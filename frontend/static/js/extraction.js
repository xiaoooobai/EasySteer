// 提取功能模块

// Extraction status polling interval ID
let extractionStatusInterval = null;

// Update extraction method options
export function updateExtractionMethodOptions() {
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

// Display extraction response
export function showExtractResponse(data) {
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
        resultHTML += `<p><strong>${window.t('status_label')}:</strong> ${data.message || window.t('extraction_complete')}</p>`;
        if (data.output_path) {
            resultHTML += `<p><strong>${window.t('output_file_label')}:</strong> <code>${data.output_path}</code></p>`;
        }
        if (data.metadata) {
            resultHTML += `<p><strong>${window.t('metadata_label')}:</strong></p>`;
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

// Display extraction error message
export function showExtractError(message) {
    const errorDiv = document.getElementById('extractError');
    const errorContent = document.getElementById('extractErrorContent');
    const responseDiv = document.getElementById('extractResponse');
    
    errorContent.textContent = message;
    errorDiv.style.display = 'block';
    responseDiv.style.display = 'none';
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Start extraction
export async function startExtraction() {
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
        showExtractError(window.t('required_fields_error'));
        return;
    }
    
    // If SAE method, validate params path
    if (extractConfig.method === 'sae' && !extractConfig.sae_params_path) {
        showExtractError(window.t('sae_path_error'));
        return;
    }
    
    // Show loading state
    const extractButton = document.querySelector('[onclick="startExtraction()"]');
    const originalHTML = extractButton.innerHTML;
    extractButton.innerHTML = '<span class="loading"></span> ' + window.t('extracting_in_progress');
    extractButton.disabled = true;
    
    try {
        // Send request to the backend
        const apiUrl = `${window.location.protocol}//${window.location.hostname}:5000/api/extract`;
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept-Language': window.currentLanguage
            },
            body: JSON.stringify(extractConfig)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showExtractResponse(data);
        } else {
            showExtractError(data.error || window.t('extraction_failed_error'));
        }
    } catch (error) {
        console.error('Extraction error:', error);
        showExtractError(window.t('network_error'));
    } finally {
        // Restore button state
        extractButton.innerHTML = originalHTML;
        extractButton.disabled = false;
    }
}

// Create extraction progress HTML
function createExtractionProgressHTML() {
    return `
        <div class="logs-container" id="extractionLogs">
            <div class="log-entry status-entry">${window.t('initializing_extraction')}</div>
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
        allLogs.push(`<div class="log-entry status-entry success-status"><strong>✅ ${window.t('extraction_complete')}</strong></div>`);
        if (status.result.output_path) {
            allLogs.push(`<div class="log-entry">📁 ${window.t('output_file_label')}: <code>${status.result.output_path}</code></div>`);
        }
        if (status.result.layers_extracted) {
            allLogs.push(`<div class="log-entry">📊 ${window.t('layers_extracted_label')}: ${status.result.layers_extracted}</div>`);
        }
    }
    
    // If there are no logs, show waiting message
    if (allLogs.length === 0) {
        allLogs.push(`<div class="log-entry status-entry">${window.t('waiting_for_extraction')}</div>`);
    }
    
    // Update log container
    logsContainer.innerHTML = allLogs.join('');
    
    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Reset extraction form
export function resetExtractForm() {
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

// Dynamically load extraction configuration options
export async function loadExtractConfigOptions() {
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
export async function importSelectedExtractConfig() {
    const extractConfigSelect = document.getElementById('extractConfigSelect');
    const selectedConfig = extractConfigSelect.value;
    
    if (!selectedConfig) {
        showExtractError(window.t('error_select_extract_config'));
        return;
    }
    
    try {
        showExtractResponse({ message: window.t('importing_extract_config', { configName: extractConfigSelect.options[extractConfigSelect.selectedIndex].text }) });
        
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
            message: window.t('extract_import_success_message'),
            description: window.t('extract_import_success_description', { configName: extractConfigSelect.options[extractConfigSelect.selectedIndex].text })
        });
        
        // Clear selection box
        extractConfigSelect.value = '';
        
    } catch (error) {
        showExtractError(window.t('extract_import_fail_error') + ': ' + error.message);
    }
} 