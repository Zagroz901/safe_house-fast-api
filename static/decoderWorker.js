// decoderWorker.js

self.onmessage = function(event) {
    const blob = new Blob([event.data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    postMessage(url);
};
