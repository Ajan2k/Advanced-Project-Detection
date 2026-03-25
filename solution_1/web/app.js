const video = document.getElementById('videoFeed');
const canvas = document.getElementById('overlayCanvas');
const ctx = canvas.getContext('2d');
let session;

// Initialize the ONNX Runtime session using the WebAssembly execution provider
async function initModel() {
    try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        session = await ort.InferenceSession.create('./models/best.onnx', {
            executionProviders: ['wasm'] // Can be upgraded to 'webgpu' for modern browser support
        });
        console.log("NMS-Free Model loaded successfully.");
        processVideo();
    } catch (e) {
        console.error("Failed to load model: ", e);
    }
}

// Change this to 1024 AFTER you retrain your model in Colab using the updated notebook!
const MODEL_RES = 1024; 

// Extract the current video frame and format it to a [1, 3, MODEL_RES, MODEL_RES] Float32Array tensor
function preprocessFrame() {
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = MODEL_RES;
    offscreenCanvas.height = MODEL_RES;
    const offCtx = offscreenCanvas.getContext('2d');
    
    // Draw and resize the video frame
    offCtx.drawImage(video, 0, 0, MODEL_RES, MODEL_RES);
    const imageData = offCtx.getImageData(0, 0, MODEL_RES, MODEL_RES).data;
    
    const float32Data = new Float32Array(3 * MODEL_RES * MODEL_RES);
    
    // Standardize to CHW (Channel, Height, Width) format and normalize pixel values to [0, 1]
    for (let i = 0; i < MODEL_RES * MODEL_RES; i++) {
        float32Data[i] = imageData[i * 4] / 255.0;               // R
        float32Data[MODEL_RES * MODEL_RES + i] = imageData[i * 4 + 1] / 255.0;   // G
        float32Data[2 * MODEL_RES * MODEL_RES + i] = imageData[i * 4 + 2] / 255.0; // B
    }
    return new ort.Tensor('float32', float32Data, [1, 3, MODEL_RES, MODEL_RES]);
}

// Run inference and render bounding boxes
async function processVideo() {
    if (video.paused || video.ended) {
        requestAnimationFrame(processVideo);
        return;
    }

    const inputTensor = preprocessFrame();
    
    // Execute the network
    const results = await session.run({ images: inputTensor });
    const output = results.output0.data; // Output shape is [1, 300, 6] for YOLOv10
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Scale factors to map MODEL_RESxMODEL_RES network output back to the original video dimensions
    const scaleX = canvas.width / MODEL_RES;
    const scaleY = canvas.height / MODEL_RES;

    // Because the model is NMS-free, we simply iterate through the direct output tensor
    // Each prediction is 6 elements: [x_min, y_min, x_max, y_max, confidence, class_id]
    for (let i = 0; i < output.length; i += 6) {
        const confidence = output[i + 4];
        
        // Filter by confidence threshold
        if (confidence > 0.5) {
            const xMin = output[i] * scaleX;
            const yMin = output[i + 1] * scaleY;
            const xMax = output[i + 2] * scaleX;
            const yMax = output[i + 3] * scaleY;
            
            // Render the prediction
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 3;
            ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
            
            ctx.fillStyle = '#00FF00';
            ctx.font = '16px Arial';
            ctx.fillText(`Tire ${(confidence * 100).toFixed(1)}%`, xMin, yMin > 20 ? yMin - 5 : 20);
        }
    }
    
    requestAnimationFrame(processVideo);
}

video.addEventListener('play', () => {
    if (!session) initModel();
});