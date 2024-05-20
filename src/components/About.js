import React from 'react';

const About = () => {
    return (
        <div>
            <div className="container mt-5">
                {/* <h2 className="display-4 mb-4">About WLASL</h2>
            <p>
                WLASL, acronym for World Level American Sign Language, represents a pioneering initiative aimed at
                advancing the recognition of dynamic sign language. Boasting a robust database, WLASL is designed to
                interpret over 2000 commonly used American Sign Language words. This extensive repertoire makes it a
                transformative tool, facilitating seamless communication within the vibrant sign language community.
            </p> */}

                <h2 className="display-4 mt-5 mb-4">How to Use</h2>
                <p>
                    Unlocking the potential of WLASL is easy by navigating to the "Sign Language Recognition" page. Users
                    can input videos through uploading, leverage their webcam, or provide a video URL. A simple click on the
                    "Predict" button yields precise interpretations based on recognized sign language gestures.
                    Furthermore, the integration of the Text-to-Speech functionality not only enhances accessibility but also provides valuable voice support, contributing to a more refined and engaging user experience.
                </p>

                <h2 className="display-4 mt-5 mb-4">Meet the Developers from VJTI</h2>
                <div className="card-deck">

                    {developers.map((developer, index) => (
                        <div key={index} className="card">
                            <div className="card-body">
                                <h5 className="card-title">{developer.name}</h5>
                                <p className="card-text">{developer.branch} - Final Year</p>
                                <p className="card-text">Email Id -{developer.email} | Reg No - {developer.reg} </p>
                                {/* <p className="card-text">VJTI College</p> */}
                            </div>
                        </div>
                    ))}
                </div>

                <br></br>
            </div>
            <br></br>
        </div>
    );
};

const developers = [
    { name: 'Avishkar Chavle', branch: 'CE', reg: '201070069', email: 'aschavle_b20@ce.vjti.ac.in' },
    { name: 'Pratham Lokhande', branch: 'CE', reg: '201070081', email: 'palokhande_d20@ce.vjti.ac.in' },
    { name: 'Chaitali Chaudhari', branch: 'CE', reg: '201071060', email: 'cnchaudhari_b20@ce.vjti.ac.in' },
    { name: 'Grishma Barule', branch: 'CE', reg: '201071062', email: 'gvbarule_b20@ce.vjti.ac.in' },
];

export default About;
