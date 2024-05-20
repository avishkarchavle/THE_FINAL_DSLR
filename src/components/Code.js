// import React, { useRef, useState, useEffect } from 'react';
// import * as tf from '@tensorflow/tfjs';
// import '@tensorflow/tfjs';
// import '@tensorflow/tfjs-node';

// const SignLanguageRecognition = () => {
//     const [i3dModel, setI3DModel] = useState(null);
//     const [videoInput, setVideoInput] = useState(null);
//     const [prediction, setPrediction] = useState('');
//     const [textToSpeechInput, setTextToSpeechInput] = useState('');
//     const [isSpeaking, setIsSpeaking] = useState(false);

//     const videoRef = useRef(null);
//     const speechSynthesisRef = useRef(window.speechSynthesis);

//     useEffect(() => {
//         // Load the I3D model
//         async function loadModel() {
//             const model = await tf.loadGraphModel('path/to/i3d/model.json');
//             setI3DModel(model);
//         }

//         loadModel();
//     }, []);

//     const handleVideoChange = (event) => {
//         const file = event.target.files[0];
//         setVideoInput(file);
//     };

//     const handlePredict = async () => {
//         if (i3dModel && videoInput) {
//             const video = document.createElement('video');
//             video.src = URL.createObjectURL(videoInput);
//             video.crossOrigin = 'anonymous';
//             video.load();

//             const frames = await captureVideoFrames(video);
//             const inputTensor = preprocessFrames(frames);

//             const predictions = i3dModel.predict(inputTensor);
//             const predictedClass = predictions.argMax(1).dataSync()[0];

//             setPrediction(`Prediction: ${predictedClass}`);
//         }
//     };

//     const captureVideoFrames = async (video) => {
//         const frames = [];
//         const canvas = document.createElement('canvas');
//         const context = canvas.getContext('2d');
//         const frameCount = 16;

//         for (let i = 0; i < frameCount; i++) {
//             context.drawImage(video, 0, 0, canvas.width, canvas.height);
//             const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
//             frames.push(tf.browser.fromPixels(imageData));
//         }

//         return frames;
//     };

//     const preprocessFrames = (frames) => {
//         const processedFrames = frames.map(frame => {
//             return tf.image.resizeBilinear(frame, [224, 224])
//                 .toFloat()
//                 .div(tf.scalar(255))
//                 .expandDims();
//         });

//         return tf.concat(processedFrames, 0);
//     };

//     const convertToSpeech = () => {
//         if (textToSpeechInput.trim() !== '') {
//             const utterance = new SpeechSynthesisUtterance(textToSpeechInput);
//             speechSynthesisRef.current.speak(utterance);
//             setIsSpeaking(true);
//         }
//     };

//     useEffect(() => {
//         return () => {
//             if (isSpeaking) {
//                 speechSynthesisRef.current.cancel();
//                 setIsSpeaking(false);
//             }
//         };
//     }, [isSpeaking]);

//     return (
//         <div className="container mt-5">
//             <h2 className="display-4 mb-4">Sign Language Recognition</h2>
//             <div className="mb-3">
//                 <label>Upload Video:</label>
//                 <input type="file" accept="video/*" onChange={handleVideoChange} />
//             </div>
//             <div className="mb-3">
//                 <button className="btn btn-success" onClick={handlePredict}>
//                     Predict
//                 </button>
//             </div>
//             <div className="mb-3">
//                 <p>{prediction}</p>
//             </div>
//             <div className="mb-3">
//                 <textarea
//                     value={textToSpeechInput}
//                     onChange={(e) => setTextToSpeechInput(e.target.value)}
//                     placeholder="Enter text for Text-to-Speech"
//                 />
//                 <br />
//                 <button className="btn btn-primary" onClick={convertToSpeech}>
//                     Convert to Speech
//                 </button>
//             </div>
//         </div>
//     );
// };
//will be useful after model 
// export default SignLanguageRecognition;
