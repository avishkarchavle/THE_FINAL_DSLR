import React, { useRef, useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css'; // Import Bootstrap CSS

const VideoRecorder = () => {
    const videoRef = useRef(null);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [isRecording, setIsRecording] = useState(false);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoRef.current.srcObject = stream;
            videoRef.current.play();

            const recorder = new MediaRecorder(stream);
            setMediaRecorder(recorder);
            const chunks = [];

            recorder.ondataavailable = (event) => {
                chunks.push(event.data);
            };

            recorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/mp4' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                document.body.appendChild(a);
                a.style = 'display: none';
                a.href = url;
                a.download = 'recording.mp4';
                a.click();
                window.URL.revokeObjectURL(url);
            };

            recorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error('Error accessing webcam:', error.name, error.message);
        }
    };

    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            setIsRecording(false);
        }
    };

    return (
        <div style={{ margin: '20px', padding: '20px', border: '1px solid #ccc', borderRadius: '5px', backgroundColor: '#f9f9f9' }}>
            <h2>Video Recorder</h2>
            <div>
                {isRecording ? (
                    <button style={{ padding: '10px 20px', margin: '10px', fontSize: '16px', cursor: 'pointer', border: 'none', borderRadius: '5px', backgroundColor: '#dc3545', color: '#fff' }} onClick={stopRecording}>Stop Recording</button>
                ) : (
                    <button style={{ padding: '10px 20px', margin: '10px', fontSize: '16px', cursor: 'pointer', border: 'none', borderRadius: '5px', backgroundColor: '#007bff', color: '#fff' }} onClick={startRecording}>Start Recording</button>
                )}
            </div>
            <div>
                <video ref={videoRef} width="640" height="480" autoPlay playsInline muted style={{ marginTop: '20px' }} />
            </div>
        </div>
    );
};

export default VideoRecorder;
