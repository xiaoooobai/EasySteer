// EasySteer - 主JavaScript文件
// 导入所有模块

// 页面切换和UI初始化
import { switchPage, switchInferenceMode, initializeUI } from './ui.js';

// 推理功能
import { submitConfiguration, importSelectedConfig, resetForm, showResponse, showError } from './inference.js';

// 多向量控制功能
import { addVectorConfig, switchVectorTab, submitMultiConfiguration, importSelectedMultiConfig, resetMultiForm, 
         updateFilePath, updateScale, updateScaleSlider, updateAlgorithm, updateLayers, updatePositions, 
         updatePrefillTokens, updatePrefillPositions, updateGenerateTokens, updateNormalize, 
         removeVectorConfig, editVectorConfig } from './multi-vector.js';

// 训练功能
import { startTraining, importSelectedTrainConfig, resetTrainForm, initTrainingInterface } from './training.js';

// 提取功能
import { startExtraction, importSelectedExtractConfig, resetExtractForm, updateExtractionMethodOptions, initExtractionInterface } from './extraction.js';

// SAE 探索功能
import { initSaeExplore, searchSaeFeatures, resetSaeForm, showSaeError, switchSaeMode, getFeatureDetailsByIndex } from './sae-explore.js';

// 聊天功能
import { initChat, sendChatMessage, selectPreset, resetChat, resetSettings } from './chat.js';

// 工具函数
import { parseListInput, escapeHtml, showStatus, restartBackend } from './utils.js';

// 导出所有函数到全局作用域，确保与原始代码兼容
window.switchPage = switchPage;
window.switchInferenceMode = switchInferenceMode;
window.initializeUI = initializeUI;
window.submitConfiguration = submitConfiguration;
window.importSelectedConfig = importSelectedConfig;
window.resetForm = resetForm;
window.showResponse = showResponse;
window.showError = showError;

window.addVectorConfig = addVectorConfig;
window.switchVectorTab = switchVectorTab;
window.submitMultiConfiguration = submitMultiConfiguration;
window.importSelectedMultiConfig = importSelectedMultiConfig;
window.resetMultiForm = resetMultiForm;
window.updateFilePath = updateFilePath;
window.updateScale = updateScale;
window.updateScaleSlider = updateScaleSlider;
window.updateAlgorithm = updateAlgorithm;
window.updateLayers = updateLayers;
window.updatePositions = updatePositions;
window.updatePrefillTokens = updatePrefillTokens;
window.updatePrefillPositions = updatePrefillPositions;
window.updateGenerateTokens = updateGenerateTokens;
window.updateNormalize = updateNormalize;
window.removeVectorConfig = removeVectorConfig;
window.editVectorConfig = editVectorConfig;

window.startTraining = startTraining;
window.importSelectedTrainConfig = importSelectedTrainConfig;
window.resetTrainForm = resetTrainForm;
window.initTrainingInterface = initTrainingInterface;

window.startExtraction = startExtraction;
window.importSelectedExtractConfig = importSelectedExtractConfig;
window.resetExtractForm = resetExtractForm;
window.updateExtractionMethodOptions = updateExtractionMethodOptions;
window.initExtractionInterface = initExtractionInterface;

// 导出 SAE 探索功能
window.searchSaeFeatures = searchSaeFeatures;
window.resetSaeForm = resetSaeForm;
window.initSaeExplore = initSaeExplore;
window.switchSaeMode = switchSaeMode;
window.getFeatureDetailsByIndex = getFeatureDetailsByIndex;
// We need to manually export this since it's defined inside sae-explore.js with window.getFeatureDetails
window.getFeatureDetails = window.getFeatureDetails || function(){};

// 导出聊天功能
window.initChat = initChat;
window.sendChatMessage = sendChatMessage;
window.resetChat = resetChat;
window.resetSettings = resetSettings;
window.selectPreset = selectPreset;


window.parseListInput = parseListInput;
window.escapeHtml = escapeHtml;
window.showStatus = showStatus;
window.restartBackend = restartBackend;

// 监听HTML模块加载完成事件
function checkModulesLoaded() {
    // 检查所有模块是否已加载
    const inferenceModule = document.getElementById('inference-module');
    const trainingModule = document.getElementById('training-module');
    const extractionModule = document.getElementById('extraction-module');
    const chatModule = document.getElementById('chat-module');
    
    if (inferenceModule.children.length > 0 && 
        trainingModule.children.length > 0 && 
        extractionModule.children.length > 0 &&
        chatModule.children.length > 0) {
        
        console.log('All HTML modules loaded, initializing UI...');
        
        // 初始化UI
        initializeUI();
        
        // 激活第一个页面
        const inferencePage = document.getElementById('inference-page');
        if (inferencePage) {
            inferencePage.classList.add('active');
        }
        
        // 初始化 SAE 探索功能
        initSaeExplore();
        
        // 初始化训练界面
        initTrainingInterface();
        
        // 初始化提取界面
        initExtractionInterface();
    } else {
        // 如果模块尚未加载完成，稍后再检查
        setTimeout(checkModulesLoaded, 100);
    }
}

// 当DOM加载完成后开始检查模块是否已加载
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded, checking modules...');
    // 开始检查模块是否已加载
    setTimeout(checkModulesLoaded, 100);
}); 