// app/static/script.js

window.onload = function() {
    const videoInput = document.getElementById('videoInput');
    const uploadButton = document.getElementById('uploadButton');
    const videoCanvas = document.getElementById('videoCanvas');
    const ctx = videoCanvas.getContext('2d');

    let websocket;
    let reconnectInterval = 5000; // 5 seconds

    function connectWebSocket() {
        console.log('Connecting WebSocket...');
        websocket = new WebSocket('ws://127.0.0.1:8000/ws/video');
        websocket.binaryType = 'arraybuffer';

        websocket.onopen = () => {
            console.log('WebSocket connection opened.');
        };

        websocket.onmessage = (event) => {
            if (event.data === '{"type": "ping"}') {
                console.log('Ping received from server.');
                return;
            }
            console.log('Message received from server.');
            const blob = new Blob([event.data], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
                ctx.drawImage(img, 0, 0, videoCanvas.width, videoCanvas.height);
                URL.revokeObjectURL(url);
            };
            img.src = url;
        };

        let reconnectAttempts = 0;

        websocket.onclose = (event) => {
            console.log('WebSocket connection closed:', event.code, event.reason);
            if (event.code !== 1000 && reconnectAttempts < 5) { // Limit reconnection attempts
                console.log('Attempting to reconnect...');
                setTimeout(() => {
                    connectWebSocket();
                    reconnectAttempts++;
                }, Math.min(5000 * 2 ** reconnectAttempts, 30000)); // Exponential back-off
            }
        };
        
        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            websocket.close(); // Ensure closure on error to trigger reconnection logic if needed
        };
        
    }

    uploadButton.addEventListener('click', () => {
        const file = videoInput.files[0];
        if (!file) {
            alert('Please select a video file first.');
            return;
        }
        console.log('Video file selected:', file.name);
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            connectWebSocket();
            websocket.onopen = () => {
                console.log('WebSocket connection opened.');
                uploadVideo(file);
            };
        } else {
            uploadVideo(file);
        }
    });

    function uploadVideo(file) {
        if (websocket.readyState === WebSocket.OPEN) {
            console.log('WebSocket is open. Sending video frames...');
            readAndSendVideo(file);
        } else {
            websocket.onopen = () => {
                console.log('WebSocket connection opened.');
                readAndSendVideo(file);
            };
        }
    }

    function readAndSendVideo(file) {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
    
        video.addEventListener('loadeddata', () => {
            console.log('Video loaded.');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            
            let frameCount = 0;
            const frameSkip = 5; // Skip every 2 frames, adjust as needed
    
            video.play();
    
            video.addEventListener('play', function () {
                console.log('Video playing...');
                const fps = 10;
                function step() {
                    if (video.paused || video.ended) {
                        console.log('Video playback ended.');
                        return;
                    }
                    
                    // if (frameCount % frameSkip === 0) {
                        context.drawImage(video, 0, 0, canvas.width, canvas.height);
                        canvas.toBlob((blob) => {
                            blob.arrayBuffer().then((buffer) => {
                                console.log('Sending video frame to server...');
                                if (websocket.readyState === WebSocket.OPEN) {
                                    websocket.send(buffer);
                                }
                            }, 'image/jpeg', 0.5);
                        });
                    // }
    
                    frameCount++;
                    setTimeout(step, 1000 / fps);
                }
                step();
            });
        });
    }
    
    // Initial WebSocket connection
    connectWebSocket();
};