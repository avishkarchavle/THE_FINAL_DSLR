import React from 'react'
import SignLanguageRecognition from './SignLanguageRecognition'

describe('<SignLanguageRecognition />', () => {
  it('renders', () => {
    // see: https://on.cypress.io/mounting-react
    cy.mount(<SignLanguageRecognition />)
  })
})