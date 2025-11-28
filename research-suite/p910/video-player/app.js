// ITU-T P.910 Video Quality Assessment Player
// Implements ACR (Absolute Category Rating) methodology

// Configuration loading
let APP_CONFIG = null;

async function loadConfiguration() {
    try {
        const response = await fetch('./config.json');
        if (!response.ok) {
            throw new Error(`Failed to load config.json: ${response.statusText}`);
        }
        const config = await response.json();

        // Auto-scan videos directory
        const videoDir = config.videos.directory;
        const dirResponse = await fetch(videoDir);
        if (!dirResponse.ok) {
            throw new Error('Failed to access videos directory');
        }

        const html = await dirResponse.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const links = doc.querySelectorAll('a');

        const videoFiles = [];
        let testVideoFile = null;
        const videoExtensions = ['.mp4', '.webm', '.ogg', '.mov'];

        links.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href !== '../') {
                const lower = href.toLowerCase();
                if (videoExtensions.some(ext => lower.endsWith(ext))) {
                    // Separate test video from assessment videos
                    if (lower === 'test.mp4') {
                        testVideoFile = href;
                    } else {
                        videoFiles.push(href);
                    }
                }
            }
        });

        config.videos.files = videoFiles;
        config.videos.testFile = testVideoFile;
        console.log(`Auto-scanned ${videoFiles.length} assessment videos`);
        if (testVideoFile) {
            console.log(`Test video found: ${testVideoFile}`);
        }

        return config;
    } catch (error) {
        console.error('Configuration loading error:', error);
        throw new Error(
            'Unable to load configuration or scan videos. ' +
            'Please ensure config.json exists and videos directory contains video files.'
        );
    }
}

function resolveVideoPath(configDirectory, filename, configBaseUrl) {
    const basePath = new URL('./', configBaseUrl).href;
    const directory = configDirectory.endsWith('/')
        ? configDirectory
        : configDirectory + '/';
    return new URL(directory + filename, basePath).href;
}

class VideoAssessment {
    constructor(config) {
        // Store configuration
        this.config = config;
        this.configBaseUrl = window.location.href;

        // Validate configuration
        const errors = this.validateConfig(config);
        if (errors.length > 0) {
            throw new Error('Configuration validation failed:\n' + errors.join('\n'));
        }

        // Extract settings from config
        this.greyScreenDuration = config.study?.greyScreenDuration || 2000;
        this.videoDirectory = config.videos.directory;
        this.videoFiles = config.videos.files || [];
        this.totalVideos = this.videoFiles.length;

        // State
        this.participantId = '';
        this.currentVideoIndex = 0;
        this.videoSequence = [];
        this.ratings = [];
        this.startTime = null;
        this.currentVideoStartTime = null;

        // Test mode state
        this.isTestMode = false;
        this.testVideoCompleted = false;
        this.hasSeenTestVideo = false;

        // DOM Elements
        this.welcomeScreen = document.getElementById('welcome-screen');
        this.videoScreen = document.getElementById('video-screen');
        this.ratingScreen = document.getElementById('rating-screen');
        this.completionScreen = document.getElementById('completion-screen');
        this.videoPlayer = document.getElementById('video-player');
        this.loadingIndicator = document.getElementById('loading-indicator');
        this.greyScreen = document.getElementById('grey-screen');

        this.initializeEventListeners();
    }

    validateConfig(config) {
        const errors = [];

        if (!config.videos?.directory) {
            errors.push('videos.directory is required in configuration');
        }

        if (config.videos?.files && !Array.isArray(config.videos.files)) {
            errors.push('videos.files must be an array');
        }

        if (!config.videos?.files || config.videos.files.length === 0) {
            errors.push('No video files found in directory');
        }

        if (config.study?.greyScreenDuration !== undefined) {
            if (typeof config.study.greyScreenDuration !== 'number') {
                errors.push('study.greyScreenDuration must be a number (milliseconds)');
            } else if (config.study.greyScreenDuration < 0) {
                errors.push('study.greyScreenDuration must be non-negative');
            }
        }

        return errors;
    }

    initializeEventListeners() {
        // Start button
        document.getElementById('start-btn').addEventListener('click', () => this.startAssessment());

        // Enter key on participant ID
        document.getElementById('participant-id').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.startAssessment();
        });

        // Rating buttons
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const rating = parseInt(e.currentTarget.dataset.rating);
                this.submitRating(rating);
            });
        });

        // Video player events
        this.videoPlayer.addEventListener('loadeddata', () => this.onVideoLoaded());
        this.videoPlayer.addEventListener('ended', () => this.onVideoEnded());
        this.videoPlayer.addEventListener('error', (e) => this.onVideoError(e));

        // Download and restart buttons
        document.getElementById('download-btn').addEventListener('click', () => this.downloadResults());
        document.getElementById('restart-btn').addEventListener('click', () => this.restart());

        // Confirmation modal buttons
        document.getElementById('test-confirm-yes').addEventListener('click', () => {
            this.handleTestConfirmation(true);
        });

        document.getElementById('test-confirm-no').addEventListener('click', () => {
            this.handleTestConfirmation(false);
        });

        // Test video start button
        document.getElementById('start-test-video-btn').addEventListener('click', () => {
            this.startTestVideo();
        });
    }

    startAssessment() {
        const participantInput = document.getElementById('participant-id');
        this.participantId = participantInput.value.trim();

        if (!this.participantId) {
            alert('Please enter a participant ID');
            participantInput.focus();
            return;
        }

        // Validate test video exists
        if (!this.config.videos.testFile) {
            alert('Error: test.mp4 not found in videos directory. Please add it before starting.');
            return;
        }

        this.startTime = new Date();

        // Enter test mode
        this.isTestMode = true;
        this.hasSeenTestVideo = true;

        // Show test instructions instead of loading video
        this.showScreen('test-instructions-screen');
    }

    startTestVideo() {
        this.showScreen('video-screen');
        this.loadTestVideo();
    }

    generateVideoSequence() {
        // Create array of video objects from config
        const videos = this.videoFiles.map((filename, index) => ({
            index: index + 1,
            filename: filename
        }));

        // Shuffle using Fisher-Yates algorithm (ITU-T P.910 requirement)
        this.videoSequence = this.shuffleArray(videos);

        console.log('Video sequence generated:', this.videoSequence);
        console.log(`Total videos: ${this.videoSequence.length}`);
    }

    shuffleArray(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    loadTestVideo() {
        const testVideoPath = resolveVideoPath(
            this.videoDirectory,
            this.config.videos.testFile,
            this.configBaseUrl
        );

        // Update progress bar to show "Test Video"
        document.getElementById('current-video').textContent = 'Test';
        document.getElementById('total-videos').textContent = 'Video';

        const progressBar = document.querySelector('.progress-bar');
        const videoMessage = document.getElementById('video-message');

        // Remove any inline styles that might be hiding them
        progressBar.style.cssText = '';
        videoMessage.style.cssText = '';

        // Ensure they are visible with full opacity
        progressBar.style.visibility = 'visible';
        progressBar.style.opacity = '1';
        videoMessage.style.visibility = 'visible';
        videoMessage.style.opacity = '1';
        videoMessage.textContent = 'Test video - please familiarize yourself with the interface';

        // Reset progress fill
        document.getElementById('progress-fill').style.width = '0%';

        // IMPORTANT: Ensure video is completely hidden during loading
        this.videoPlayer.style.opacity = '0';
        this.videoPlayer.style.visibility = 'hidden';

        this.loadingIndicator.style.display = 'block';
        this.hideGreyScreen();

        // Load video
        this.videoPlayer.src = testVideoPath;
        this.videoPlayer.load();

        console.log('Loading test video:', testVideoPath);
    }

    loadNextVideo() {
        // Guard against being called in test mode
        if (this.isTestMode) {
            console.error('loadNextVideo() called in test mode - this should not happen');
            return;
        }

        if (this.currentVideoIndex >= this.totalVideos) {
            this.completeAssessment();
            return;
        }

        const videoInfo = this.videoSequence[this.currentVideoIndex];
        const videoPath = resolveVideoPath(
            this.videoDirectory,
            videoInfo.filename,
            this.configBaseUrl
        );

        // Update progress
        document.getElementById('current-video').textContent = this.currentVideoIndex + 1;
        document.getElementById('total-videos').textContent = this.totalVideos;

        const progressPercent = ((this.currentVideoIndex) / this.totalVideos) * 100;
        document.getElementById('progress-fill').style.width = progressPercent + '%';

        // IMPORTANT: Reset and show progress bar while loading, but HIDE message (no longer test mode)
        const progressBar = document.querySelector('.progress-bar');
        const videoMessage = document.getElementById('video-message');

        // Remove any inline styles that might be hiding progress bar
        progressBar.style.cssText = '';

        // Ensure progress bar is visible with full opacity
        progressBar.style.visibility = 'visible';
        progressBar.style.opacity = '1';

        // HIDE the video message completely during assessment phase
        videoMessage.style.display = 'none';

        // IMPORTANT: Ensure video is completely hidden during loading
        this.videoPlayer.style.opacity = '0';
        this.videoPlayer.style.visibility = 'hidden';

        this.loadingIndicator.style.display = 'block';
        this.hideGreyScreen();

        // Load video
        this.videoPlayer.src = videoPath;
        this.videoPlayer.load();

        console.log(`Loading video ${this.currentVideoIndex + 1}:`, videoPath);
    }

    onVideoLoaded() {
        console.log('Video loaded successfully');

        // Show grey screen FIRST, instantly without transition
        // This prevents any flash of the video element
        this.greyScreen.style.transition = 'none';  // Disable transition
        this.greyScreen.classList.add('active');

        // Force reflow to apply instant grey screen
        void this.greyScreen.offsetHeight;

        // Re-enable transition for the hide animation later
        this.greyScreen.style.transition = '';

        // Ensure video stays completely hidden
        this.videoPlayer.style.opacity = '0';
        this.videoPlayer.style.visibility = 'hidden';

        // Now hide the loading indicator after grey screen is shown
        this.loadingIndicator.style.display = 'none';

        // Keep progress bar and message visible during grey screen
        // They will be hidden when video actually starts playing

        // Wait for grey screen duration, then show video and play
        setTimeout(() => {
            // Start fade-out of progress bar and message
            const progressBar = document.querySelector('.progress-bar');
            const videoMessage = document.getElementById('video-message');
            progressBar.style.opacity = '0';
            videoMessage.style.opacity = '0';

            // After fade-out completes, hide them completely and start video
            setTimeout(() => {
                progressBar.style.visibility = 'hidden';
                videoMessage.style.visibility = 'hidden';

                this.hideGreyScreen();

                // Now make video visible
                this.videoPlayer.style.visibility = 'visible';
                this.videoPlayer.style.opacity = '1';

                this.currentVideoStartTime = new Date();
                this.videoPlayer.play().catch(err => {
                    console.error('Autoplay failed:', err);
                    // Show message only if autoplay fails
                    videoMessage.style.display = 'block';
                    videoMessage.style.visibility = 'visible';
                    videoMessage.style.opacity = '1';
                    videoMessage.textContent = 'Click the video to play';
                });
            }, 800); // Match the CSS transition duration (0.8s)
        }, this.greyScreenDuration);
    }

    onVideoEnded() {
        console.log('Video playback ended');

        // Show grey screen after video
        this.showGreyScreen();

        // Wait for grey screen duration, then show rating screen
        setTimeout(() => {
            this.hideGreyScreen();
            this.showScreen('rating-screen');
        }, this.greyScreenDuration);
    }

    onVideoError(e) {
        console.error('Video error:', e);

        if (this.isTestMode) {
            alert('Error loading test video: test.mp4\n\nPlease ensure the video file exists and is properly formatted.');
        } else {
            const videoInfo = this.videoSequence[this.currentVideoIndex];
            alert(`Error loading video: ${videoInfo.filename}\n\nPlease ensure the video file exists in the videos folder.`);
        }
    }

    submitRating(rating) {
        const currentTime = new Date();
        const responseTime = this.currentVideoStartTime
            ? (currentTime - this.currentVideoStartTime) / 1000
            : 0;

        // Test mode branch - do not save rating
        if (this.isTestMode) {
            console.log('Test rating submitted (not saved):', rating);
            this.testVideoCompleted = true;
            this.showScreen('video-screen');
            this.showConfirmationModal();
            return;
        }

        // Assessment mode - save rating
        const videoInfo = this.videoSequence[this.currentVideoIndex];
        const ratingData = {
            videoIndex: this.currentVideoIndex + 1,
            filename: videoInfo.filename,
            rating: rating,
            timestamp: currentTime.toISOString(),
            responseTime: responseTime.toFixed(2)
        };

        this.ratings.push(ratingData);
        console.log('Rating submitted:', ratingData);

        // Move to next video
        this.currentVideoIndex++;
        this.showScreen('video-screen');
        this.loadNextVideo();
    }

    showConfirmationModal() {
        const modal = document.getElementById('test-confirmation-modal');
        modal.classList.add('active');
    }

    hideConfirmationModal() {
        const modal = document.getElementById('test-confirmation-modal');
        modal.classList.remove('active');
    }

    handleTestConfirmation(ready) {
        this.hideConfirmationModal();

        if (!ready) {
            // User clicked "No" - repeat test video
            console.log('User requested test repeat');
            this.showScreen('video-screen');
            this.loadTestVideo();
        } else {
            // User clicked "Yes" - start assessment
            console.log('User confirmed readiness - starting assessment');
            this.isTestMode = false;
            this.testVideoCompleted = false;

            // NOW generate the assessment sequence
            this.generateVideoSequence();
            this.currentVideoIndex = 0;

            this.showScreen('video-screen');
            this.loadNextVideo();
        }
    }

    completeAssessment() {
        const endTime = new Date();
        const totalTime = ((endTime - this.startTime) / 1000).toFixed(0);
        const minutes = Math.floor(totalTime / 60);
        const seconds = totalTime % 60;

        // Update completion screen
        document.getElementById('final-participant-id').textContent = this.participantId;
        document.getElementById('final-video-count').textContent = this.totalVideos;
        document.getElementById('completion-time').textContent = `${minutes}m ${seconds}s`;

        this.showScreen('completion-screen');
    }

    downloadResults() {
        const csv = this.generateCSV();
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;

        // Use configured filename pattern or default
        const pattern = this.config.export?.filenamePattern ||
                        'p910_assessment_{participantId}_{timestamp}.csv';
        const timestamp = Date.now();
        const filename = pattern
            .replace('{participantId}', this.participantId)
            .replace('{timestamp}', timestamp);

        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    generateCSV() {
        const headers = [
            'ParticipantID',
            'VideoIndex',
            'Filename',
            'Rating',
            'Timestamp',
            'ResponseTime_seconds'
        ];

        const rows = this.ratings.map(r => [
            this.participantId,
            r.videoIndex,
            r.filename,
            r.rating,
            r.timestamp,
            r.responseTime
        ]);

        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        return csvContent;
    }

    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        document.getElementById(screenId).classList.add('active');
    }

    showGreyScreen() {
        this.greyScreen.classList.add('active');
        console.log('Grey screen shown');
    }

    hideGreyScreen() {
        this.greyScreen.classList.remove('active');
        console.log('Grey screen hidden');
    }

    restart() {
        // Reset state
        this.participantId = '';
        this.currentVideoIndex = 0;
        this.videoSequence = [];
        this.ratings = [];
        this.startTime = null;
        this.currentVideoStartTime = null;

        // Reset UI
        document.getElementById('participant-id').value = '';
        this.videoPlayer.src = '';

        this.showScreen('welcome-screen');
    }
}

// Initialize the application
async function initializeApp() {
    try {
        console.log('Loading configuration...');
        APP_CONFIG = await loadConfiguration();
        console.log('Configuration loaded:', APP_CONFIG);

        const app = new VideoAssessment(APP_CONFIG);
        console.log('ITU-T P.910 Video Assessment Player initialized');

        // Make app globally available for debugging
        window.videoAssessmentApp = app;
    } catch (error) {
        console.error('Application initialization failed:', error);

        // Display error to user
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #f44336;
            color: white;
            padding: 30px;
            border-radius: 8px;
            max-width: 600px;
            text-align: center;
            z-index: 10000;
            font-family: Arial, sans-serif;
        `;
        errorDiv.innerHTML = `
            <h2>Configuration Error</h2>
            <p>${error.message}</p>
            <p style="margin-top: 20px; font-size: 14px;">
                Please check that config.json exists in the video-player directory
                and is properly formatted.
            </p>
        `;
        document.body.appendChild(errorDiv);
    }
}

// Start application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
