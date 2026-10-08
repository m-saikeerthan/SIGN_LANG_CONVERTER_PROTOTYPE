/**
 * ISL Detector — Web Frontend Logic
 * Supports:
 *  1. Live Real-Time Stream Mode
 *  2. Dynamic Video Input Mode (Continuous Recording, Smart Deduplication to 2-3 keyframes,
 *     Gesture Dictionary Grid, and Interactive Multi-Frame Keyframe Viewer)
 */

// ─── DOM Selector Helper ─────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ─── Screen Elements ────────────────────────────────────────────────
const screenHome     = $('screen-home');
const screenDetector = $('screen-detector');
const screenFuture   = $('screen-future');
const screenAbout    = $('screen-about');
const screenMore     = $('screen-more');

// Navigation Elements
const navBtnHome     = $('nav-btn-home');
const navBtnDetector = $('nav-btn-detector');
const navBtnFuture   = $('nav-btn-future');
const navBtnAbout    = $('nav-btn-about');
const navBtnMore     = $('nav-btn-more');
const navItems       = document.querySelectorAll('.nav-item');
const statusBadge    = $('status-badge');
const toast          = $('toast');

// Mode Switch Elements
const tabModeLive    = $('tab-mode-live');
const tabModeVideo   = $('tab-mode-video');
const viewModeLive   = $('view-mode-live');
const viewModeVideo  = $('view-mode-video');

// ─── Mode 1: Live Stream Elements ───────────────────────────────────
const video            = $('video');
const canvas           = $('canvas');
const ctx              = canvas ? canvas.getContext('2d') : null;
const cameraOverlay    = $('camera-overlay');
const annotatedOverlay = $('annotated-overlay');
const cameraStatus     = $('camera-status');
const statusDot        = $('status-dot');
const detectionStatus  = $('detection-status');
const btnCamera        = $('btn-camera');
const fpsDisplay       = $('fps-display');
const btnHandRight     = $('hand-right');
const btnHandLeft      = $('hand-left');
const btnClear         = $('btn-clear');
const predictedWord    = $('predicted-word');
const confidenceFill   = $('confidence-fill');
const confidenceText   = $('confidence-text');
const timerContainer   = $('timer-container');
const timerFill        = $('timer-fill');
const wordChips        = $('word-chips');
const outputCard       = $('output-card');
const correctedSentence = $('corrected-sentence');
const btnSpeak         = $('btn-speak');
const btnCopy          = $('btn-copy');

// Single Snapshot Modal (Live Mode)
const gestureModal = $('gesture-modal');
const modalClose   = $('modal-close');
const modalImage   = $('modal-image');
const modalWord    = $('modal-word');

// ─── Mode 2: Dynamic Video Elements ─────────────────────────────────
const recordPreviewVideo      = $('record-preview-video');
const playbackVideo           = $('playback-video');
const recordPlaceholder       = $('record-placeholder');
const recordingBadge          = $('recording-badge');
const recordingTimer          = $('recording-timer');
const btnStartRecord          = $('btn-start-record');
const btnStopRecord           = $('btn-stop-record');
const uploadDropzone          = $('upload-dropzone');
const videoFileInput          = $('video-file-input');
const videoProcessingCard     = $('video-processing-card');
const processingStatusText    = $('processing-status-text');
const videoProgressFill       = $('video-progress-fill');
const videoMetricsPill        = $('video-metrics-pill');
const videoSentencePlaceholder= $('video-sentence-placeholder');
const videoSentenceContent    = $('video-sentence-content');
const videoClickableSentence  = $('video-clickable-sentence');
const videoHindiSentence      = $('video-hindi-sentence');
const btnSpeakVideo           = $('btn-speak-video');
const btnCopyVideo            = $('btn-copy-video');
const btnClearVideo           = $('btn-clear-video');
const gestureDictionaryGrid   = $('gesture-dictionary-grid');

// Multi-Frame Modal
const multiframeModal         = $('multiframe-modal');
const multiframeModalClose    = $('multiframe-modal-close');
const modalWordName           = $('modal-word-name');
const modalWordConf           = $('modal-word-conf');
const modalWordTime           = $('modal-word-time');
const modalKeyframesGallery   = $('modal-keyframes-gallery');

// ─── State Variables ────────────────────────────────────────────────
let currentMode = 'live'; // 'live' | 'video'
let selectedHand = 'right';
let cameraActive = false;
let streamRef    = null;
let sendInterval = null;
let lastCorrectedSentence = '';
let frameCount    = 0;
let fpsStartTime  = Date.now();
let lastSentenceKey = '';
let liveWordSnapshots = {};

// Video Recording State
let mediaRecorder = null;
let recordedChunks = [];
let recordStream = null;
let recordTimerInterval = null;
let recordSeconds = 0;
let currentGestureDict = {};
let lastVideoEnglishSentence = '';

// ─── Socket.IO Setup ────────────────────────────────────────────────
const socket = io();

socket.on('connect', () => {
    if (statusBadge) {
        statusBadge.textContent = 'Connected';
        statusBadge.classList.remove('connecting');
    }
});

socket.on('disconnect', () => {
    if (statusBadge) {
        statusBadge.textContent = 'Disconnected';
        statusBadge.classList.add('connecting');
    }
});

// ─── Top-Level Navigation ───────────────────────────────────────────
function switchScreen(screenId, activeNavBtn) {
    screenHome.classList.add('hidden');
    screenDetector.classList.add('hidden');
    if (screenFuture) screenFuture.classList.add('hidden');
    if (screenAbout) screenAbout.classList.add('hidden');
    if (screenMore) screenMore.classList.add('hidden');
    
    const targetScreen = $(screenId);
    if (targetScreen) targetScreen.classList.remove('hidden');

    navItems.forEach(item => item.classList.remove('active'));
    if (activeNavBtn) activeNavBtn.classList.add('active');

    if (screenId !== 'screen-detector') {
        if (cameraActive) stopCamera();
        if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
    }
}

if (navBtnHome) navBtnHome.addEventListener('click', e => { e.preventDefault(); switchScreen('screen-home', navBtnHome); });
if (navBtnDetector) navBtnDetector.addEventListener('click', e => { e.preventDefault(); switchScreen('screen-detector', navBtnDetector); });
if (navBtnFuture) navBtnFuture.addEventListener('click', e => { e.preventDefault(); switchScreen('screen-future', navBtnFuture); });
if (navBtnAbout) navBtnAbout.addEventListener('click', e => { e.preventDefault(); switchScreen('screen-about', navBtnAbout); });
if (navBtnMore) navBtnMore.addEventListener('click', e => { e.preventDefault(); switchScreen('screen-more', navBtnMore); });

const btnHeroStart = $('btn-hero-start');
if (btnHeroStart) {
    btnHeroStart.addEventListener('click', () => {
        switchScreen('screen-detector', navBtnDetector);
        if (currentMode === 'live' && !cameraActive) {
            startCamera();
        }
    });
}

const btnHeroFuture = $('btn-hero-future');
if (btnHeroFuture) {
    btnHeroFuture.addEventListener('click', () => {
        switchScreen('screen-future', navBtnFuture);
    });
}

// ─── Mode Switching (Live Stream vs Dynamic Video) ───────────────────
function setDetectionMode(mode) {
    currentMode = mode;
    if (mode === 'live') {
        tabModeLive.classList.add('active');
        tabModeVideo.classList.remove('active');
        viewModeLive.classList.remove('hidden');
        viewModeVideo.classList.add('hidden');
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        }
    } else {
        tabModeVideo.classList.add('active');
        tabModeLive.classList.remove('active');
        viewModeVideo.classList.remove('hidden');
        viewModeLive.classList.add('hidden');
        if (cameraActive) {
            stopCamera();
        }
    }
}

if (tabModeLive) tabModeLive.addEventListener('click', () => setDetectionMode('live'));
if (tabModeVideo) tabModeVideo.addEventListener('click', () => setDetectionMode('video'));

// ─── Hand Preference Toggle (Live Mode) ─────────────────────────────
function setHandPreference(hand) {
    selectedHand = hand;
    if (btnHandRight) btnHandRight.classList.toggle('active', hand === 'right');
    if (btnHandLeft) btnHandLeft.classList.toggle('active', hand === 'left');
    if (video) video.style.transform = hand === 'left' ? 'scaleX(1)' : 'scaleX(-1)';
    if (annotatedOverlay) annotatedOverlay.style.transform = hand === 'left' ? 'scaleX(1)' : 'scaleX(-1)';
}

if (btnHandRight) btnHandRight.addEventListener('click', () => setHandPreference('right'));
if (btnHandLeft) btnHandLeft.addEventListener('click', () => setHandPreference('left'));

// ═════════════════════════════════════════════════════════════════════
// MODE 1: LIVE REAL-TIME STREAM LOGIC
// ═════════════════════════════════════════════════════════════════════

socket.on('prediction', data => {
    if (currentMode !== 'live') return;

    // FPS
    frameCount++;
    const elapsed = (Date.now() - fpsStartTime) / 1000;
    if (elapsed >= 1 && fpsDisplay) {
        fpsDisplay.textContent = `${Math.round(frameCount / elapsed)} FPS`;
        frameCount = 0;
        fpsStartTime = Date.now();
    }

    // Annotated frame overlay
    if (data.annotated_frame && cameraActive && annotatedOverlay) {
        annotatedOverlay.src = data.annotated_frame;
        annotatedOverlay.style.display = 'block';
    }

    // Status dot
    if (cameraStatus && statusDot && detectionStatus) {
        cameraStatus.style.display = 'flex';
        statusDot.className = 'status-dot';
        if (data.hand_detected) {
            statusDot.classList.add(data.status);
            detectionStatus.textContent = capitalize(data.status);
        } else {
            detectionStatus.textContent = 'No Hand';
        }
    }

    // Predicted word
    if (predictedWord) {
        if (data.word) {
            predictedWord.textContent = data.word;
            if (data.status === 'confirmed') predictedWord.style.color = 'var(--green)';
            else if (data.status === 'holding') predictedWord.style.color = 'var(--accent-purple)';
            else predictedWord.style.color = 'var(--text-primary)';
        } else {
            predictedWord.textContent = '—';
            predictedWord.style.color = 'var(--text-primary)';
        }
    }

    // Confidence bar
    if (confidenceFill && confidenceText) {
        const conf = Math.round(data.confidence * 100);
        confidenceFill.style.width = conf + '%';
        confidenceText.textContent = conf + '%';
        if (data.status === 'holding') confidenceFill.style.background = 'var(--accent-purple)';
        else if (data.status === 'confirmed') confidenceFill.style.background = 'var(--green)';
        else confidenceFill.style.background = 'var(--text-primary)';
    }

    // Capture snapshot thumbnail on confirmation
    if (data.snapshot && data.status === 'confirmed' && data.word) {
        liveWordSnapshots[data.word] = data.snapshot;
    }

    // Update word chips
    const sentenceKey = (data.sentence || []).join('|');
    if (sentenceKey !== lastSentenceKey) {
        lastSentenceKey = sentenceKey;
        updateLiveWordChips(data.sentence);
    }

    // Inactivity timer
    if (timerContainer && timerFill) {
        if (!data.hand_detected && data.no_hand_seconds > 0 && data.sentence && data.sentence.length > 0) {
            timerContainer.style.display = 'block';
            const progress = Math.min(data.no_hand_seconds / 3, 1) * 100;
            timerFill.style.width = progress + '%';
        } else {
            timerContainer.style.display = 'none';
        }
    }
});

socket.on('sentence_complete', data => {
    if (currentMode !== 'live') return;
    lastCorrectedSentence = data.corrected;
    if (outputCard) outputCard.style.display = 'block';
    buildClickableSentence(data.corrected, data.sentence || [], correctedSentence, liveWordSnapshots);
    if (timerContainer) timerContainer.style.display = 'none';
    speak(data.corrected);
});

function updateLiveWordChips(words) {
    if (!wordChips) return;
    if (!words || words.length === 0) {
        wordChips.innerHTML = '<span class="words-placeholder">Words will appear here...</span>';
        return;
    }
    wordChips.innerHTML = '';
    words.forEach(w => {
        const chip = document.createElement('div');
        chip.className = 'word-chip';
        
        const snap = liveWordSnapshots[w];
        if (snap) {
            const img = document.createElement('img');
            img.src = snap;
            img.alt = w;
            chip.appendChild(img);

            chip.addEventListener('click', () => {
                if (modalImage && modalWord && gestureModal) {
                    modalImage.src = snap;
                    modalWord.textContent = w;
                    gestureModal.style.display = 'flex';
                }
            });
        }
        
        const span = document.createElement('span');
        span.textContent = w;
        chip.appendChild(span);
        wordChips.appendChild(chip);
    });
}

// Live Camera Controls
if (btnCamera) {
    btnCamera.addEventListener('click', () => {
        if (cameraActive) stopCamera();
        else startCamera();
    });
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        streamRef = stream;
        video.srcObject = stream;
        video.play().catch(e => console.warn("video.play() error:", e));
        cameraActive = true;

        setHandPreference(selectedHand);

        if (cameraOverlay) cameraOverlay.style.display = 'none';
        btnCamera.textContent = '⏹ Stop Camera';
        btnCamera.classList.add('active');

        video.onloadedmetadata = () => {
            if (canvas) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }
        };

        sendInterval = setInterval(captureAndSendLiveFrame, 100);
    } catch (err) {
        console.error('Camera error:', err);
        showToast('Camera access denied or unavailable.');
    }
}

function stopCamera() {
    if (streamRef) streamRef.getTracks().forEach(t => t.stop());
    if (sendInterval) clearInterval(sendInterval);
    cameraActive = false;
    if (video) video.srcObject = null;
    if (annotatedOverlay) annotatedOverlay.style.display = 'none';
    if (cameraOverlay) cameraOverlay.style.display = 'flex';
    if (cameraStatus) cameraStatus.style.display = 'none';
    if (btnCamera) {
        btnCamera.innerHTML = '📷 Start Camera';
        btnCamera.classList.remove('active');
    }
}

function captureAndSendLiveFrame() {
    if (!cameraActive || !video.videoWidth || !ctx || !canvas) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    socket.emit('frame', { image: canvas.toDataURL('image/jpeg', 0.7) });
}

if (btnClear) {
    btnClear.addEventListener('click', () => {
        socket.emit('clear_sentence');
        lastSentenceKey = '';
        updateLiveWordChips([]);
        if (outputCard) outputCard.style.display = 'none';
        if (predictedWord) {
            predictedWord.textContent = '—';
            predictedWord.style.color = 'var(--text-primary)';
        }
        if (confidenceFill) confidenceFill.style.width = '0%';
        if (confidenceText) confidenceText.textContent = '0%';
    });
}

if (btnCopy) {
    btnCopy.addEventListener('click', () => {
        if (lastCorrectedSentence) {
            navigator.clipboard.writeText(lastCorrectedSentence).then(() => showToast('Copied to clipboard!'));
        }
    });
}

if (btnSpeak) {
    btnSpeak.addEventListener('click', () => {
        if (lastCorrectedSentence) speak(lastCorrectedSentence);
    });
}


// ═════════════════════════════════════════════════════════════════════
// MODE 2: DYNAMIC VIDEO INPUT (CONTINUOUS RECORD & DEDUPLICATION)
// ═════════════════════════════════════════════════════════════════════

if (btnStartRecord) {
    btnStartRecord.addEventListener('click', startRecording);
}

if (btnStopRecord) {
    btnStopRecord.addEventListener('click', stopRecording);
}

async function startRecording() {
    try {
        recordedChunks = [];
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        });
        recordStream = stream;
        recordPreviewVideo.srcObject = stream;
        recordPreviewVideo.style.display = 'block';
        if (playbackVideo) playbackVideo.style.display = 'none';
        if (recordPlaceholder) recordPlaceholder.style.display = 'none';

        // Select suitable mimeType supported by browser
        let mimeType = 'video/webm;codecs=vp8';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = 'video/webm';
        }

        mediaRecorder = new MediaRecorder(stream, { mimeType });
        mediaRecorder.ondataavailable = e => {
            if (e.data && e.data.size > 0) {
                recordedChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: mimeType });
            // Show preview video
            if (playbackVideo) {
                playbackVideo.src = URL.createObjectURL(blob);
                playbackVideo.style.display = 'block';
                recordPreviewVideo.style.display = 'none';
            }
            processRecordedVideoBlob(blob);
        };

        mediaRecorder.start(200); // Collect in 200ms slices
        btnStartRecord.style.display = 'none';
        btnStopRecord.style.display = 'inline-flex';
        
        // Recording timer
        recordSeconds = 0;
        if (recordingTimer) recordingTimer.textContent = '00:00';
        if (recordingBadge) recordingBadge.style.display = 'inline-flex';
        recordTimerInterval = setInterval(() => {
            recordSeconds++;
            const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
            const secs = String(recordSeconds % 60).padStart(2, '0');
            if (recordingTimer) recordingTimer.textContent = `${mins}:${secs}`;
        }, 1000);

        showToast('Recording started. Perform your signs continuously!');
    } catch (err) {
        console.error('Recording error:', err);
        showToast('Failed to start camera for recording.');
    }
}

function stopRecording() {
    if (recordTimerInterval) clearInterval(recordTimerInterval);
    if (recordingBadge) recordingBadge.style.display = 'none';
    if (btnStopRecord) btnStopRecord.style.display = 'none';
    if (btnStartRecord) btnStartRecord.style.display = 'inline-flex';

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    if (recordStream) {
        recordStream.getTracks().forEach(t => t.stop());
    }
}

// File Upload Handling
if (uploadDropzone && videoFileInput) {
    uploadDropzone.addEventListener('click', () => videoFileInput.click());

    uploadDropzone.addEventListener('dragover', e => {
        e.preventDefault();
        uploadDropzone.classList.add('drag-active');
    });

    uploadDropzone.addEventListener('dragleave', () => {
        uploadDropzone.classList.remove('drag-active');
    });

    uploadDropzone.addEventListener('drop', e => {
        e.preventDefault();
        uploadDropzone.classList.remove('drag-active');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleVideoFile(e.dataTransfer.files[0]);
        }
    });

    videoFileInput.addEventListener('change', e => {
        if (e.target.files && e.target.files.length > 0) {
            handleVideoFile(e.target.files[0]);
        }
    });
}

function handleVideoFile(file) {
    if (!file.type.startsWith('video/')) {
        showToast('Please select a valid video file (.mp4, .webm, etc.)');
        return;
    }
    if (playbackVideo) {
        playbackVideo.src = URL.createObjectURL(file);
        playbackVideo.style.display = 'block';
        if (recordPreviewVideo) recordPreviewVideo.style.display = 'none';
        if (recordPlaceholder) recordPlaceholder.style.display = 'none';
    }
    processRecordedVideoBlob(file);
}

// Process Video through Server API
async function processRecordedVideoBlob(blob) {
    if (videoProcessingCard) videoProcessingCard.style.display = 'block';
    if (processingStatusText) processingStatusText.textContent = 'Extracting frames & removing duplicate poses...';

    const formData = new FormData();
    formData.append('video', blob, 'recording.webm');

    try {
        const response = await fetch('/api/process_video', {
            method: 'POST',
            body: formData
        });

        const res = await response.json();
        if (videoProcessingCard) videoProcessingCard.style.display = 'none';

        if (res.status === 'success' && res.data) {
            renderDynamicVideoResults(res.data);
            showToast('Video processed successfully!');
        } else {
            showToast('Error processing video: ' + (res.message || 'Unknown error'));
        }
    } catch (err) {
        console.error('API Error:', err);
        if (videoProcessingCard) videoProcessingCard.style.display = 'none';
        showToast('Network error during video processing.');
    }
}

// Render Results & Gesture Dictionary
function renderDynamicVideoResults(data) {
    currentGestureDict = data.gesture_dict || {};
    lastVideoEnglishSentence = data.corrected_sentence || '';

    // Show Sentence Box
    if (videoSentencePlaceholder) videoSentencePlaceholder.style.display = 'none';
    if (videoSentenceContent) videoSentenceContent.style.display = 'block';
    if (videoMetricsPill) {
        videoMetricsPill.textContent = `${data.total_frames} frames • ${data.processed_keyframes_count} keyframes • ${data.execution_time_sec}s`;
        videoMetricsPill.style.display = 'inline-flex';
    }

    // Build Clickable English Sentence
    if (videoClickableSentence) {
        videoClickableSentence.innerHTML = '';
        const words = (data.corrected_sentence || '').split(/\s+/);
        words.forEach((w, i) => {
            const span = document.createElement('span');
            span.textContent = w;
            const matchKey = findDictMatch(w, currentGestureDict);
            if (matchKey) {
                span.className = 'sentence-word clickable';
                span.title = `Click to see 2-3 keyframes for "${matchKey}"`;
                span.addEventListener('click', () => openMultiFrameModal(matchKey));
            } else {
                span.className = 'sentence-word';
            }
            videoClickableSentence.appendChild(span);
            if (i < words.length - 1) {
                videoClickableSentence.appendChild(document.createTextNode(' '));
            }
        });
    }

    // Hindi Translation
    if (videoHindiSentence) {
        videoHindiSentence.textContent = data.hindi_sentence || '—';
    }

    // Speak English sentence automatically
    if (data.corrected_sentence) {
        speak(data.corrected_sentence);
    }

    // Render Detected Gesture Dictionary Cards Grid
    renderGestureDictionaryGrid(data.gesture_dict || {});
}

function renderGestureDictionaryGrid(dict) {
    if (!gestureDictionaryGrid) return;
    const entries = Object.entries(dict);
    if (entries.length === 0) {
        gestureDictionaryGrid.innerHTML = `
            <div class="dictionary-empty">
                No distinct sign gestures detected. Try holding each gesture steadily for ~1 second.
            </div>
        `;
        return;
    }

    gestureDictionaryGrid.innerHTML = '';
    entries.forEach(([word, info]) => {
        const card = document.createElement('div');
        card.className = 'dict-card';

        const confPct = Math.round((info.confidence || 0) * 100);
        const timeStr = info.timestamps && info.timestamps.length > 0
            ? `${info.timestamps[0]}s - ${info.timestamps[info.timestamps.length - 1]}s`
            : '';
        const keyframeCount = (info.frames || []).length;
        const mainThumbnail = (info.frames && info.frames.length > 0) ? info.frames[Math.floor(info.frames.length / 2)] : '';

        card.innerHTML = `
            <div class="dict-card-thumb-wrap">
                <img src="${mainThumbnail}" alt="${word}" class="dict-card-thumb">
                <span class="dict-frame-badge">🖼️ ${keyframeCount} Keyframes</span>
            </div>
            <div class="dict-card-content">
                <div class="dict-card-header">
                    <span class="dict-card-word">${word}</span>
                    <span class="dict-card-conf">${confPct}%</span>
                </div>
                <div class="dict-card-time">⏱ ${timeStr}</div>
                <button class="btn-view-keyframes">🔍 View Captured Frames</button>
            </div>
        `;

        card.addEventListener('click', () => openMultiFrameModal(word));
        gestureDictionaryGrid.appendChild(card);
    });
}

// ─── Multi-Frame Keyframe Modal Logic ────────────────────────────────
function openMultiFrameModal(word) {
    const info = currentGestureDict[word];
    if (!info) return;

    if (modalWordName) modalWordName.textContent = word;
    if (modalWordConf) modalWordConf.textContent = `${Math.round(info.confidence * 100)}% Accuracy`;
    if (modalWordTime) {
        const timeStr = info.timestamps && info.timestamps.length > 0
            ? `${info.timestamps[0]}s - ${info.timestamps[info.timestamps.length - 1]}s`
            : '';
        modalWordTime.textContent = `⏱ Video Time: ${timeStr}`;
    }

    if (modalKeyframesGallery) {
        modalKeyframesGallery.innerHTML = '';
        const frames = info.frames || [];
        frames.forEach((frameB64, idx) => {
            const frameCard = document.createElement('div');
            frameCard.className = 'keyframe-item';
            
            const timestamp = (info.timestamps && info.timestamps[idx] !== undefined) ? `${info.timestamps[idx]}s` : `Frame ${idx + 1}`;
            const phaseLabel = idx === 0 ? 'Onset Phase' : idx === 1 ? 'Apex / Hold Phase' : 'Resolution Phase';

            frameCard.innerHTML = `
                <div class="keyframe-img-wrap">
                    <img src="${frameB64}" alt="${word} Keyframe ${idx + 1}" class="keyframe-img">
                    <span class="keyframe-phase-tag">${phaseLabel}</span>
                </div>
                <div class="keyframe-meta">
                    <span class="keyframe-tag">Keyframe ${idx + 1}</span>
                    <span class="keyframe-timestamp">⏱ ${timestamp}</span>
                </div>
            `;
            modalKeyframesGallery.appendChild(frameCard);
        });
    }

    if (multiframeModal) multiframeModal.style.display = 'flex';
}

if (multiframeModalClose) {
    multiframeModalClose.addEventListener('click', () => {
        if (multiframeModal) multiframeModal.style.display = 'none';
    });
}

if (multiframeModal) {
    multiframeModal.addEventListener('click', e => {
        if (e.target === multiframeModal) {
            multiframeModal.style.display = 'none';
        }
    });
}

// Sentence Button Actions (Dynamic Video Mode)
if (btnSpeakVideo) {
    btnSpeakVideo.addEventListener('click', () => {
        if (lastVideoEnglishSentence) speak(lastVideoEnglishSentence);
    });
}

if (btnCopyVideo) {
    btnCopyVideo.addEventListener('click', () => {
        if (lastVideoEnglishSentence) {
            navigator.clipboard.writeText(lastVideoEnglishSentence).then(() => showToast('Sentence copied to clipboard!'));
        }
    });
}

if (btnClearVideo) {
    btnClearVideo.addEventListener('click', () => {
        currentGestureDict = {};
        lastVideoEnglishSentence = '';
        if (videoSentencePlaceholder) videoSentencePlaceholder.style.display = 'block';
        if (videoSentenceContent) videoSentenceContent.style.display = 'none';
        if (videoMetricsPill) videoMetricsPill.style.display = 'none';
        if (gestureDictionaryGrid) {
            gestureDictionaryGrid.innerHTML = `
                <div class="dictionary-empty">
                    No gestures processed yet. Record video to see dictionary with captured frames.
                </div>
            `;
        }
        if (playbackVideo) playbackVideo.style.display = 'none';
        if (recordPlaceholder) recordPlaceholder.style.display = 'flex';
    });
}

// ─── General Helper Functions ───────────────────────────────────────
function buildClickableSentence(text, rawWords, container, snapshotsMap) {
    if (!container) return;
    container.innerHTML = '';
    const words = (text || '').split(/\s+/);

    words.forEach((w, i) => {
        const span = document.createElement('span');
        span.textContent = w;

        const matchKey = findSnapshotMatch(w, rawWords, snapshotsMap);
        if (matchKey && snapshotsMap[matchKey]) {
            span.className = 'sentence-word clickable';
            span.title = `Click to see gesture for "${matchKey}"`;
            span.addEventListener('click', () => {
                if (modalImage && modalWord && gestureModal) {
                    modalImage.src = snapshotsMap[matchKey];
                    modalWord.textContent = matchKey;
                    gestureModal.style.display = 'flex';
                }
            });
        } else {
            span.className = 'sentence-word';
        }

        container.appendChild(span);
        if (i < words.length - 1) {
            container.appendChild(document.createTextNode(' '));
        }
    });
}

function findSnapshotMatch(word, rawWords, snapshotsMap) {
    const clean = word.toUpperCase().replace(/[^A-Z]/g, '');
    if (snapshotsMap[clean]) return clean;
    for (const raw of rawWords) {
        if (snapshotsMap[raw] && raw.toUpperCase().startsWith(clean.slice(0, 3))) {
            return raw;
        }
    }
    return null;
}

function findDictMatch(word, dict) {
    const clean = word.toUpperCase().replace(/[^A-Z]/g, '');
    if (dict[clean]) return clean;
    for (const raw of Object.keys(dict)) {
        if (raw.toUpperCase().startsWith(clean.slice(0, 3)) || clean.startsWith(raw.slice(0, 3))) {
            return raw;
        }
    }
    return null;
}

function speak(text) {
    if (!('speechSynthesis' in window) || !text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
}

function showToast(msg) {
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2800);
}

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Modal closing for Live mode modal
if (modalClose) {
    modalClose.addEventListener('click', () => {
        if (gestureModal) gestureModal.style.display = 'none';
    });
}
if (gestureModal) {
    gestureModal.addEventListener('click', e => {
        if (e.target === gestureModal) gestureModal.style.display = 'none';
    });
}
