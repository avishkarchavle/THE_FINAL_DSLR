const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
const port = process.env.PORT || 5000;

app.use(bodyParser.json());

// Replace this with your actual machine learning code
const predictSignLanguage = async (videoFrame) => {
    // Placeholder code: You need to replace this with your actual prediction logic
    // For simplicity, I'm just returning a random class as a placeholder
    const classes = ['Sign1', 'Sign2', 'Sign3'];
    return classes[Math.floor(Math.random() * classes.length)];
};

app.post('/predict', async (req, res) => {
    const { videoFrame } = req.body;

    try {
        const prediction = await predictSignLanguage(videoFrame);
        res.json({ prediction });
    } catch (error) {
        console.error('Error predicting:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
