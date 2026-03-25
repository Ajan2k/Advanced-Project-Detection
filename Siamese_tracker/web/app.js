const video = document.getElementById('videoFeed');
const canvas = document.getElementById('uiCanvas');
const ctx = canvas.getContext('2d');

let trackerSession;
let templateTensor = null; // Store the initial cropped image instead of features
let currentBox = null; // [x, y, width, height]
let isDrawing = false, startX, startY;

// Load the combined OSTrack model using WebAssembly
async function loadModels() {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    trackerSession = await ort.InferenceSession.create('./models/ostrack_merged.onnx', { executionProviders: ['wasm'] });
    console.log("OSTrack model loaded.");
}
loadModels();

// Handle user drawing the initialization box
canvas.addEventListener('mousedown', (e) => {
    if (!video.paused) return;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    isDrawing = true;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'red'; ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
});

canvas.addEventListener('mouseup', async (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    const rect = canvas.getBoundingClientRect();
    
    // Store the initial bounding box
    currentBox = {
        x: Math.min(startX, e.clientX - rect.left),
        y: Math.min(startY, e.clientY - rect.top),
        w: Math.abs(e.clientX - rect.left - startX),
        h: Math.abs(e.clientY - rect.top - startY)
    };
    
    // Extract template and run Stage 1
    await initializeTracker();
});

// Helper: Crop an image area and convert to Float32 Tensor [1, 3, H, W]
function getTensorFromCrop(x, y, w, h, targetSize) {
    const offCanvas = document.createElement('canvas');
    offCanvas.width = targetSize; offCanvas.height = targetSize;
    const offCtx = offCanvas.getContext('2d');
    
    // Scale UI canvas coordinates (640x480) to the video's intrinsic resolution (e.g. 1920x1080)
    const scaleX = video.videoWidth / canvas.width;
    const scaleY = video.videoHeight / canvas.height;
    
    // Draw cropped region from video onto scaled canvas
    offCtx.drawImage(
        video, 
        x * scaleX, y * scaleY, w * scaleX, h * scaleY, 
        0, 0, targetSize, targetSize
    );
    const imgData = offCtx.getImageData(0, 0, targetSize, targetSize).data;
    
    const floatData = new Float32Array(3 * targetSize * targetSize);
    
    // Standard ImageNet Normalization used by OSTrack (ViT backbone)
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    for (let i = 0; i < targetSize * targetSize; i++) {
        floatData[i] = ((imgData[i * 4] / 255.0) - mean[0]) / std[0]; // R
        floatData[targetSize * targetSize + i] = ((imgData[i * 4 + 1] / 255.0) - mean[1]) / std[1]; // G
        floatData[2 * targetSize * targetSize + i] = ((imgData[i * 4 + 2] / 255.0) - mean[2]) / std[2]; // B
    }
    return new ort.Tensor('float32', floatData, [1, 3, targetSize, targetSize]);
}

// Stage 1: Encode the target
async function initializeTracker() {
    // 128x128 is a standard template size. OSTrack requires the template to be padded 
    // by exactly 2x the target size (just like the search region) so it sees background context!
    const padX = currentBox.w * 0.5;
    const padY = currentBox.h * 0.5;
    const tempX = Math.max(0, currentBox.x - padX);
    const tempY = Math.max(0, currentBox.y - padY);
    const tempW = currentBox.w + (padX * 2);
    const tempH = currentBox.h + (padY * 2);

    templateTensor = getTensorFromCrop(tempX, tempY, tempW, tempH, 128);
    console.log("Initial template image cropped and stored.");
}

// Stage 2: Track target across frames
async function trackFrame() {
    if (video.paused || video.ended || !templateTensor) return;

    // Define search region (usually 2x the size of the target, centered on last known location)
    const searchPadX = currentBox.w * 0.5;
    const searchPadY = currentBox.h * 0.5;
    const searchX = Math.max(0, currentBox.x - searchPadX);
    const searchY = Math.max(0, currentBox.y - searchPadY);
    const searchW = currentBox.w + (searchPadX * 2);
    const searchH = currentBox.h + (searchPadY * 2);

    const searchTensor = getTensorFromCrop(searchX, searchY, searchW, searchH, 256);
    
    // Run tracking network with current frame AND stored template
    const inputs = { template_img: templateTensor, search_img: searchTensor };
    const results = await trackerSession.run(inputs);
    
    // Model outputs relative coordinates within the 256x256 search region.
    // The export script outputs 'pred_boxes' (cx, cy, w, h normalized)
    const pred = results.pred_boxes.data; // [x, y, w, h] normalized 0-1
    
    currentBox.x = searchX + (pred[0] * searchW) - ((pred[2] * searchW) / 2);
    currentBox.y = searchY + (pred[1] * searchH) - ((pred[3] * searchH) / 2);
    
    // Clamp width and height so they can never mathematically collapse to 0
    currentBox.w = Math.max(10, pred[2] * searchW);
    currentBox.h = Math.max(10, pred[3] * searchH);

    // Render
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#00ff00'; ctx.lineWidth = 3;
    ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);

    requestAnimationFrame(trackFrame);
}

video.addEventListener('play', () => {
    if (templateTensor) trackFrame();
});

document.getElementById('playBtn').addEventListener('click', () => {
    video.play();
});

document.getElementById('pauseBtn').addEventListener('click', () => {
    video.pause();
});