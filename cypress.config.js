const { defineConfig } = require('cypress');

module.exports = defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',  // React app URL
    setupNodeEvents(on, config) {
      // implement node event listeners here
    },
    fixturesFolder: 'cypress/fixtures',
    supportFile: 'cypress/support/index.js',  // Ensure this file exists
    specPattern: 'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}'
  },
});
