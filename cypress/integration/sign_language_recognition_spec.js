// describe('Sign Language Recognition', () => {
//   it('Uploads a video file and predicts', () => {
//     cy.visit('http://localhost:3000')
    
//     // Simulate file upload
//     const fileName = '00336.mp4';
//     cy.fixture(fileName).then(fileContent => {
//       cy.get('input[type="file"]').upload({ fileContent, fileName, mimeType: 'video/mp4' });
//     });

//     // Click the Predict button
//     cy.get('.btn-success').click();

//     // Wait for prediction to appear
//     cy.get('p').contains('Prediction:').should('exist');
//   })
// })
describe('Sign Language Recognition', () => {
  it('Starts and stops webcam recording', () => {
    cy.visit('http://localhost:3000')
    
    // Start webcam recording
    cy.get('.btn-primary').click();

    // Wait for a few seconds (simulate recording time)
    cy.wait(5000);

    // Stop webcam recording
    cy.get('.btn-danger').click();

    // Ensure the video input is not null
    cy.window().its('videoInput').should('not.be.null');
  })
})

