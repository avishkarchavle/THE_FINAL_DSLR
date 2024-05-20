import React from 'react';
import signLanguageImage from './images/sign_language.jpg';

const Home = () => {
    return (
        <div className="container mt-5">
            <div className="row">
                <div className="col-md-6">
                    <h1 className="display-4 mb-4">Dynamic Sign Language Recognition</h1>
                    <p className="lead">
                        Welcome to our innovative platform for dynamic sign language recognition. We use advanced
                        algorithms to interpret sign language gestures, making communication accessible to everyone.
                    </p>
                    <p>
                        Whether you are a part of the Deaf community or a learner of sign language, our system can help
                        you understand and interpret various sign language expressions in real-time.
                    </p>
                    <p>
                        Get started by navigating to the "Sign Language Recognition" page, where you can upload videos,
                        use your webcam, or input a video URL to predict sign language gestures.
                    </p>
                    <a className="btn btn-primary mt-3" href="/sign-language-recognition" role="button">
                        Try Sign Language Recognition
                    </a>
                </div>
                <div className="col-md-6">
                    <img src={signLanguageImage} alt="Sign Language" className="img-fluid rounded" />
                </div>
            </div>

            <hr className="my-5" />

            <div className="row">
                <div className="col-md-6">
                    <h2 className="display-4 mb-4">About WLASL</h2>
                    <p>
                        WLASL, which stands for World Level American Sign Language, is a comprehensive project designed to
                        recognize dynamic sign language. With a vast database of over 2000 common American Sign Language
                        words, WLASL is a powerful tool for communication.
                    </p>
                    <p>
                        To learn more about WLASL and how to use our platform, visit the "About" page.
                    </p>
                    <a className="btn btn-info mt-3" href="/about" role="button">
                        Learn More About WLASL
                    </a>
                    <br></br>
                </div>
                <div className="col-md-6">
                    {/* Add additional image or content here */}
                </div>
            </div>
            <br></br>
        </div>
    );
};

export default Home;
