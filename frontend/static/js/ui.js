// UI相关功能模块

// 导入其他模块的依赖
import { loadConfigOptions } from './inference.js';
import { loadTrainConfigOptions } from './training.js';
import { loadExtractConfigOptions } from './extraction.js';
import { loadMultiConfigOptions } from './multi-vector.js';

// 更新聊天强度滑块值显示
export function updateSteerStrengthDisplay(value) {
    const displayValue = parseFloat(value).toFixed(1);
    document.querySelectorAll('.steer-slider-labels span')[1].textContent = displayValue;
}

// Page switching functionality
export function switchPage(pageType) {
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
    } else if (pageType === 'chat') {
        document.getElementById('chat-page').classList.add('active');
        document.querySelector('[onclick="switchPage(\'chat\')"]').classList.add('active');
        // 初始化聊天界面
        initChat();
    }
}

// Inference sub-page switching functionality
export function switchInferenceMode(mode) {
    // Hide all inference modes
    document.querySelectorAll('.inference-mode').forEach(inferenceMode => {
        inferenceMode.classList.remove('active');
        inferenceMode.style.display = 'none';
    });
    
    // Remove active class from all sub-navigation buttons
    document.querySelectorAll('.sub-nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show the selected inference mode
    if (mode === 'single') {
        document.getElementById('single-vector-mode').classList.add('active');
        document.getElementById('single-vector-mode').style.display = 'block';
        document.querySelector('[onclick="switchInferenceMode(\'single\')"]').classList.add('active');
    } else if (mode === 'multi') {
        document.getElementById('multi-vector-mode').classList.add('active');
        document.getElementById('multi-vector-mode').style.display = 'block';
        document.querySelector('[onclick="switchInferenceMode(\'multi\')"]').classList.add('active');
        loadMultiConfigOptions();
    } else if (mode === 'sae-explore') {
        document.getElementById('sae-explore-mode').classList.add('active');
        document.getElementById('sae-explore-mode').style.display = 'block';
        document.querySelector('[onclick="switchInferenceMode(\'sae-explore\')"]').classList.add('active');
    }
}

// Initialize UI components
export function initializeUI() {
    // Load configuration options
    loadConfigOptions();
    loadTrainConfigOptions();
    loadExtractConfigOptions();

    // Initialize synchronization for sliders and number inputs for single vector mode
    const scaleSlider = document.getElementById('scaleSlider');
    const scaleInput = document.getElementById('scale');
    const temperatureSlider = document.getElementById('temperatureSlider');
    const temperatureInput = document.getElementById('temperature');
    const repetitionPenaltySlider = document.getElementById('repetitionPenaltySlider');
    const repetitionPenaltyInput = document.getElementById('repetitionPenalty');

    if(scaleSlider && scaleInput) {
        scaleSlider.addEventListener('input', () => scaleInput.value = scaleSlider.value);
        scaleInput.addEventListener('input', () => scaleSlider.value = scaleInput.value);
    }
    if(temperatureSlider && temperatureInput) {
        temperatureSlider.addEventListener('input', () => temperatureInput.value = temperatureSlider.value);
        temperatureInput.addEventListener('input', () => temperatureSlider.value = temperatureInput.value);
    }
    if(repetitionPenaltySlider && repetitionPenaltyInput) {
        repetitionPenaltySlider.addEventListener('input', () => repetitionPenaltyInput.value = repetitionPenaltySlider.value);
        repetitionPenaltyInput.addEventListener('input', () => repetitionPenaltySlider.value = repetitionPenaltyInput.value);
    }

    // Handle file selection for single vector mode
    const fileInput = document.getElementById('fileInput');
    if(fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('localPath').value = file.name;
            }
        });
    }
    
    // Synchronize multi-mode sliders
    const multiTemperatureSlider = document.getElementById('multiTemperatureSlider');
    const multiTemperatureInput = document.getElementById('multiTemperature');
    const multiRepetitionPenaltySlider = document.getElementById('multiRepetitionPenaltySlider');
    const multiRepetitionPenaltyInput = document.getElementById('multiRepetitionPenalty');
    
    if (multiTemperatureSlider && multiTemperatureInput) {
        multiTemperatureSlider.addEventListener('input', () => multiTemperatureInput.value = multiTemperatureSlider.value);
        multiTemperatureInput.addEventListener('input', () => multiTemperatureSlider.value = multiTemperatureInput.value);
    }
    
    if (multiRepetitionPenaltySlider && multiRepetitionPenaltyInput) {
        multiRepetitionPenaltySlider.addEventListener('input', () => multiRepetitionPenaltyInput.value = multiRepetitionPenaltySlider.value);
        multiRepetitionPenaltyInput.addEventListener('input', () => multiRepetitionPenaltySlider.value = multiRepetitionPenaltyInput.value);
    }
    
    // Handle SAE combination mode change
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
    
    // Add focus effect for all input fields with help-text
    const inputs = document.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            const helpText = this.closest('.form-group')?.querySelector('.help-text');
            if (helpText) {
                helpText.style.color = 'var(--primary-color)';
            }
        });
        
        input.addEventListener('blur', function() {
            const helpText = this.closest('.form-group')?.querySelector('.help-text');
            if (helpText) {
                helpText.style.color = 'var(--text-secondary)';
            }
        });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            // Determine which form is active
            if (document.getElementById('single-vector-mode')?.classList.contains('active')) {
                window.submitConfiguration();
            } else if (document.getElementById('multi-vector-mode')?.classList.contains('active')) {
                window.submitMultiConfiguration();
            } else if (document.getElementById('sae-explore-mode')?.classList.contains('active')) {
                window.searchSaeFeatures();
            } else if (document.getElementById('chat-page')?.classList.contains('active')) {
                window.sendChatMessage();
            }
        }
    });

    // Call the function to set the initial state of extraction method options
    window.updateExtractionMethodOptions();
} 