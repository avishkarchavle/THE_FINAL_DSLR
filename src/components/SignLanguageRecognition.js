// import React, { useRef, useState } from 'react';
// import axios from 'axios';

// const SignLanguageRecognition = () => {
//     const [videoInput, setVideoInput] = useState(null);
//     const [prediction, setPrediction] = useState('');
//     const [isSpeaking, setIsSpeaking] = useState(false);

//     const videoRef = useRef(null);
//     const speechSynthesisRef = useRef(window.speechSynthesis);
//     const utteranceRef = useRef(null);

//     const handleVideoChange = (event) => {
//         const file = event.target.files[0];
//         setVideoInput(file);
//         console.log("Video file selected: ", file);
//     };

//     const handlePredict = async () => {
//         try {
//             const formData = new FormData();
//             if (videoInput instanceof Blob) {
//                 formData.append('video', videoInput, 'video.webm');
//                 console.log("Video blob appended to formData");
//             } else {
//                 alert('Please provide a video input');
//                 return;
//             }

//             console.log("Sending request to backend...");
//             const response = await axios.post('http://localhost:5000/predict', formData, {
//                 headers: {
//                     'Content-Type': 'multipart/form-data',
//                 },
//             });

//             console.log("Response received: ", response.data);
//             setPrediction(response.data.prediction);
//             speakText(response.data.prediction);
//         } catch (error) {
//             console.error('Error predicting:', error);
//         }
//     };

//     const speakText = (text) => {
//         utteranceRef.current = new SpeechSynthesisUtterance(text);
//         speechSynthesisRef.current.speak(utteranceRef.current);
//         setIsSpeaking(true);
//     };

//     const stopSpeaking = () => {
//         if (isSpeaking) {
//             speechSynthesisRef.current.cancel();
//             setIsSpeaking(false);
//         }
//     };

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
//             {prediction && (
//                 <div className="mt-3">
//                     <p>Prediction: {prediction}</p>
//                     <button
//                         className={`btn ${isSpeaking ? 'btn-danger' : 'btn-success'}`}
//                         onClick={() => {
//                             if (isSpeaking) {
//                                 stopSpeaking();
//                             } else {
//                                 speakText(prediction);
//                             }
//                         }}
//                     >
//                         {isSpeaking ? 'Stop Speaking' : 'Speak'}
//                     </button>
//                 </div>
//             )}
//         </div>
//     );
// };

// export default SignLanguageRecognition;
// import React, { useRef, useState } from 'react';
// import axios from 'axios';

// const SignLanguageRecognition = () => {
//     const [videoInput, setVideoInput] = useState(null);
//     const [predictions, setPredictions] = useState([]);
//     const [isSpeaking, setIsSpeaking] = useState(false);

//     const videoRef = useRef(null);
//     const speechSynthesisRef = useRef(window.speechSynthesis);
//     const utteranceRef = useRef(null);

//     const handleVideoChange = (event) => {
//         const file = event.target.files[0];
//         setVideoInput(file);
//         console.log("Video file selected: ", file);
//     };

//     const handlePredict = async () => {
//         try {
//             const formData = new FormData();
//             if (videoInput instanceof Blob) {
//                 formData.append('video', videoInput, 'video.webm');
//                 console.log("Video blob appended to formData");
//             } else {
//                 alert('Please provide a video input');
//                 return;
//             }

//             console.log("Sending request to backend...");
//             const response = await axios.post('http://localhost:5000/predict', formData, {
//                 headers: {
//                     'Content-Type': 'multipart/form-data',
//                 },
//             });

//             console.log("Response received: ", response.data);
//             setPredictions(response.data); // Update predictions with response data
//             speakPredictions(response.data); // Speak the predicted words
//         } catch (error) {
//             console.error('Error predicting:', error);
//         }
//     };

//     const speakPredictions = (predictions) => {
//         if (predictions && predictions.length > 0) {
//             predictions.forEach((prediction) => {
//                 speakText(prediction);
//             });
//         }
//     };

//     const speakText = (text) => {
//         utteranceRef.current = new SpeechSynthesisUtterance(text);
//         speechSynthesisRef.current.speak(utteranceRef.current);
//         setIsSpeaking(true);
//     };

//     const stopSpeaking = () => {
//         if (isSpeaking) {
//             speechSynthesisRef.current.cancel();
//             setIsSpeaking(false);
//         }
//     };

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
//             {predictions && predictions.length > 0 && (
//                 <div className="mt-3">
//                     <p>Predicted Words:</p>
//                     <ul>
//                         {predictions.map((word, index) => (
//                             <li key={index}>{word}</li>
//                         ))}
//                     </ul>
//                 </div>
//             )}
//         </div>
//     );
// };

// export default SignLanguageRecognition;
import React, { useRef, useState } from 'react';
import axios from 'axios';

const SignLanguageRecognition = () => {
    const [videoInput, setVideoInput] = useState(null);
    const [predictions, setPredictions] = useState([]);
    const [selectedWord, setSelectedWord] = useState('');
    const [isSpeaking, setIsSpeaking] = useState(false);

    const videoRef = useRef(null);
    const speechSynthesisRef = useRef(window.speechSynthesis);
    const utteranceRef = useRef(null);

    const handleVideoChange = (event) => {
        const file = event.target.files[0];
        setVideoInput(file);
        console.log("Video file selected: ", file);
    };

    const handlePredict = async () => {
        try {
            const formData = new FormData();
            if (videoInput instanceof Blob) {
                formData.append('video', videoInput, 'video.webm');
                console.log("Video blob appended to formData");
            } else {
                alert('Please provide a video input');
                return;
            }

            console.log("Sending request to backend...");
            const response = await axios.post('http://localhost:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            console.log("Response received: ", response.data);
            setPredictions(response.data); // Update predictions with response data
        } catch (error) {
            console.error('Error predicting:', error);
        }
    };

    const speakWord = () => {
        if (selectedWord) {
            speakText(selectedWord);
        }
    };

    const speakText = (text) => {
        utteranceRef.current = new SpeechSynthesisUtterance(text);
        speechSynthesisRef.current.speak(utteranceRef.current);
        setIsSpeaking(true);
    };

    const stopSpeaking = () => {
        if (isSpeaking) {
            speechSynthesisRef.current.cancel();
            setIsSpeaking(false);
        }
    };

    return (
        <div style={{ marginTop: '50px' }}>
            <h2 style={{ marginBottom: '20px', fontSize: '24px' }}>Sign Language Recognition</h2>
            <div style={{ marginBottom: '20px' }}>
                <label style={{ marginRight: '10px' }}>Upload Video:</label>
                <input type="file" accept="video/*" onChange={handleVideoChange} />
            </div>
            <div style={{ marginBottom: '20px' }}>
                <button style={{ backgroundColor: '#28a745', color: 'white', border: 'none', padding: '8px 16px', borderRadius: '5px' }} onClick={handlePredict}>
                    Predict
                </button>
            </div>
            {predictions && predictions.length > 0 && (
                <div style={{ marginBottom: '20px' }}>
                    <p style={{ fontSize: '18px', fontWeight: 'bold' }}>Predicted Words:</p>
                    <ul style={{ listStyleType: 'none', padding: '0' }}>
                        {predictions.map((word, index) => (
                            <li key={index} style={{ marginBottom: '5px' }}>
                                <button
                                    style={{ backgroundColor: selectedWord === word ? '#007bff' : 'transparent', color: selectedWord === word ? 'white' : '#007bff', border: 'none', cursor: 'pointer', padding: '5px 10px', borderRadius: '5px' }}
                                    onClick={() => setSelectedWord(word)}
                                >
                                    {word}
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
            <div>
                <button
                    style={{ backgroundColor: isSpeaking ? '#dc3545' : '#28a745', color: 'white', border: 'none', padding: '8px 16px', borderRadius: '5px' }}
                    onClick={isSpeaking ? stopSpeaking : speakWord}
                    disabled={!selectedWord}
                >
                    {isSpeaking ? 'Stop Speaking' : 'Speak'}
                </button>
            </div>
        </div>
    );
};

export default SignLanguageRecognition;
