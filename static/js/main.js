// ===== BRAIN TUMOR CLASSIFIER - MAIN JS =====

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const cancelBtn = document.getElementById('cancelBtn');
const loading = document.getElementById('loading');
const errorMsg = document.getElementById('errorMsg');
const resultsSection = document.getElementById('resultsSection');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

let selectedFile = null;

// ===== LOAD METRICS ON PAGE LOAD =====
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    checkPredictionState();
});

// ===== CHECK FOR ONGOING/COMPLETED PREDICTION =====
function checkPredictionState() {
    const predictionInProgress = sessionStorage.getItem('predictionInProgress');
    const lastResult = sessionStorage.getItem('lastPredictionResult');
    
    if (predictionInProgress === 'true') {
        // Show loading state with a message that prediction is ongoing
        const startTime = parseInt(sessionStorage.getItem('predictionStartTime'));
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        
        const loading = document.getElementById('loading');
        if (loading) {
            loading.style.display = 'block';
            
            // Restart animation from appropriate stage based on elapsed time
            const stages = ['stage1', 'stage2', 'stage3', 'stage4'];
            const timings = [0.5, 1.7, 3.2, 4.0]; // Cumulative seconds
            
            let currentStageIndex = 3; // Default to last stage if time has passed
            for (let i = 0; i < timings.length; i++) {
                if (elapsed < timings[i]) {
                    currentStageIndex = i;
                    break;
                }
            }
            
            // Mark completed stages
            for (let i = 0; i < currentStageIndex; i++) {
                const stage = document.getElementById(stages[i]);
                if (stage) stage.classList.add('completed');
            }
            
            // Mark current stage as active
            if (currentStageIndex < stages.length) {
                const stage = document.getElementById(stages[currentStageIndex]);
                if (stage) {
                    stage.classList.add('active');
                    animatePipelineStages(); // Continue animation
                }
            }
            
            // Add notice that user can navigate away
            const existingNotice = loading.querySelector('.nav-notice');
            if (!existingNotice) {
                const noticeDiv = document.createElement('div');
                noticeDiv.className = 'nav-notice';
                noticeDiv.style.cssText = 'text-align: center; margin-top: 20px; padding: 15px; background: rgba(102, 126, 234, 0.1); border-radius: 10px; color: #667eea; font-weight: 600;';
                noticeDiv.innerHTML = '💡 Tip: Processing continues in background. Feel free to check the <a href="/metrics" style="color: #667eea; text-decoration: underline;">Metrics page</a>!';
                loading.appendChild(noticeDiv);
            }
        }
        
    } else if (lastResult) {
        // Show the last completed result
        try {
            const data = JSON.parse(lastResult);
            const resultsSection = document.getElementById('resultsSection');
            if (resultsSection) {
                displayResults(data);
                
                // Add a "Clear Results" button
                const actionButtons = resultsSection.querySelector('.action-buttons');
                if (actionButtons && !actionButtons.querySelector('.clear-results-btn')) {
                    const clearBtn = document.createElement('button');
                    clearBtn.className = 'btn btn-secondary clear-results-btn';
                    clearBtn.innerHTML = '<span class="btn-icon">🔄</span> Clear & Analyze New';
                    clearBtn.style.marginRight = '10px';
                    clearBtn.onclick = () => {
                        sessionStorage.removeItem('lastPredictionResult');
                        location.reload();
                    };
                    actionButtons.prepend(clearBtn);
                }
            }
            
        } catch (e) {
            console.error('Error loading previous result:', e);
            sessionStorage.removeItem('lastPredictionResult');
        }
    }
}

async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        if (data.error) {
            console.error('Failed to load metrics:', data.error);
            return;
        }
        
        // Update stats bar
        document.getElementById('accuracy').textContent = data.performance.reported_accuracy;
        document.getElementById('samples').textContent = data.dataset.train_samples.toLocaleString();
        document.getElementById('speed').textContent = data.performance.avg_inference_time;
        
        // Add GPU indicator
        if (data.performance.gpu_accelerated) {
            const speedElement = document.getElementById('speed');
            speedElement.innerHTML = `${data.performance.avg_inference_time} <span style="color: #4CAF50;">⚡</span>`;
        }
        
    } catch (error) {
        console.error('Error loading metrics:', error);
        // Set fallback values (from actual training results - DenseNet121 on combined dataset)
        // Note: Total enhanced was 10,087 but combined dataset is 8,087 (notumor class filtered)
        document.getElementById('accuracy').textContent = '99.21%';
        document.getElementById('samples').textContent = '6,568';
        document.getElementById('speed').textContent = '~51ms';
    }
}

// ===== UPLOAD HANDLERS =====

// Click to upload
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image file (PNG, JPG, BMP, TIFF).');
        return;
    }
    
    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        errorMsg.style.display = 'none';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// ===== CANCEL BUTTON =====
cancelBtn.addEventListener('click', () => {
    selectedFile = null;
    previewSection.style.display = 'none';
    fileInput.value = '';
});

// ===== PIPELINE STAGE ANIMATION =====
function animatePipelineStages() {
    const stages = ['stage1', 'stage2', 'stage3', 'stage4'];
    const timings = [500, 1200, 1500, 800]; // Time for each stage in ms
    
    // Reset all stages
    stages.forEach(stageId => {
        const stage = document.getElementById(stageId);
        stage.classList.remove('active', 'completed');
    });
    
    let currentStage = 0;
    
    return new Promise((resolve) => {
        function progressToNextStage() {
            if (currentStage > 0) {
                const prevStage = document.getElementById(stages[currentStage - 1]);
                prevStage.classList.remove('active');
                prevStage.classList.add('completed');
            }
            
            if (currentStage < stages.length) {
                const stage = document.getElementById(stages[currentStage]);
                stage.classList.add('active');
                
                setTimeout(() => {
                    currentStage++;
                    progressToNextStage();
                }, timings[currentStage]);
            } else {
                // All stages complete
                setTimeout(resolve, 300);
            }
        }
        
        progressToNextStage();
    });
}

// ===== ANALYZE BUTTON =====
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Show loading state
    previewSection.style.display = 'none';
    loading.style.display = 'block';
    errorMsg.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Mark that prediction is in progress
    sessionStorage.setItem('predictionInProgress', 'true');
    sessionStorage.setItem('predictionStartTime', Date.now());
    
    // Start pipeline animation
    const animationPromise = animatePipelineStages();
    
    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Store results in session storage
        sessionStorage.setItem('lastPredictionResult', JSON.stringify(data));
        sessionStorage.removeItem('predictionInProgress');
        
        // Wait for animation to complete before showing results
        await animationPromise;
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        sessionStorage.removeItem('predictionInProgress');
        loading.style.display = 'none';
        showError(error.message);
        previewSection.style.display = 'block';
    }
});

// ===== DISPLAY RESULTS =====
function displayResults(data) {
    loading.style.display = 'none';
    resultsSection.style.display = 'block';
    
    const diagnosisCard = document.getElementById('diagnosisCard');
    const resultTitle = document.getElementById('resultTitle');
    const diagnosisBadge = document.getElementById('diagnosisBadge');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidencePercent = document.getElementById('confidencePercent');
    const highlightInfo = document.getElementById('highlightInfo');
    
    // Check if prediction is uncertain
    const isUncertain = data.is_uncertain;
    
    if (isUncertain) {
        // Style for uncertain predictions
        diagnosisCard.classList.add('uncertain');
        resultTitle.innerHTML = `⚠️ UNCERTAIN PREDICTION<br><small style="font-size: 0.5em; font-weight: 400;">Predicted: ${data.prediction} (Below ${data.confidence_threshold.toFixed(0)}% threshold)</small>`;
        diagnosisBadge.textContent = 'Requires Review';
        diagnosisBadge.style.background = 'rgba(255, 255, 255, 0.4)';
        
        // Show detailed warning
        highlightInfo.className = 'highlight-info warning-box';
        highlightInfo.innerHTML = `
            <div style="margin-bottom: 15px;">
                <strong>⚠️ ${data.warning_message}</strong>
            </div>
            <div style="font-size: 0.95rem;">
                <strong>Possible reasons:</strong>
                <ul style="margin: 10px 0 15px 20px; text-align: left;">
                    <li>Normal brain scan (no tumor detected)</li>
                    <li>Poor image quality or unusual scan type</li>
                    <li>Atypical tumor presentation</li>
                    <li>Non-brain MRI image uploaded</li>
                </ul>
                <strong>📋 Recommendation:</strong> Manual review by a qualified radiologist is required for accurate diagnosis.
            </div>
        `;
    } else {
        // Style for confident predictions
        diagnosisCard.classList.remove('uncertain');
        resultTitle.textContent = `Diagnosis: ${data.prediction}`;
        diagnosisBadge.textContent = 'High Confidence';
        diagnosisBadge.style.background = 'rgba(255, 255, 255, 0.3)';
        
        highlightInfo.className = 'highlight-info';
        highlightInfo.innerHTML = `
            <strong>🎯 Tumor Region Highlighted:</strong> ${data.highlighted_area.toFixed(2)}% of scan<br>
            <span style="font-size: 0.95rem; opacity: 0.9;">The AI has identified and highlighted the suspected tumor region in the visualization below.</span>
        `;
    }
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = data.confidence + '%';
        confidencePercent.textContent = data.confidence.toFixed(1) + '%';
        
        if (isUncertain) {
            confidenceFill.classList.add('uncertain');
        } else {
            confidenceFill.classList.remove('uncertain');
        }
    }, 100);
    
    // Display all class probabilities
    const probabilities = document.getElementById('probabilities');
    probabilities.innerHTML = '';
    
    const sortedProbs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    
    sortedProbs.forEach(([className, prob]) => {
        const isTopPrediction = className === data.prediction;
        probabilities.innerHTML += `
            <div class="prob-item" style="${isTopPrediction ? 'border: 2px solid rgba(255,255,255,0.5);' : ''}">
                <div class="prob-label">${className} ${isTopPrediction ? '⭐' : ''}</div>
                <div class="prob-value">${prob.toFixed(1)}%</div>
            </div>
        `;
    });
    
    // Set visualization image
    document.getElementById('visualizationImage').src = 
        'data:image/png;base64,' + data.visualization;
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
}

// ===== NEW ANALYSIS BUTTON =====
newAnalysisBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorMsg.style.display = 'none';
    
    // Scroll back to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ===== ERROR DISPLAY =====
function showError(message) {
    errorMsg.textContent = '❌ ' + message;
    errorMsg.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        errorMsg.style.display = 'none';
    }, 5000);
}

// ===== SMOOTH SCROLL FOR NAV LINKS =====
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
