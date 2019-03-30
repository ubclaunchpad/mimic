import React, { Component } from 'react';
import Container from 'react-bootstrap/Container'
import Row from 'react-bootstrap/Row'
import './App.css';
import TextGenerator from './TextGenerator';

class App extends Component {

  state = {
    generatedText: 'Your generated text will appear here.',
  };

  // Passed to Text Generator
  textGenerated = (text) => {
    this.setState({ generatedText: text });
  };

  render() {
    return (
      <Container className="App">
        <div className="centered">
          <h1>Mimic ML Text Generator</h1>
          <p>Generate text from an initial phrase. Pick from pre-trained models &amp; different ML algorithms!</p>
        </div>
        <Row className="text-generator-input">
          <TextGenerator textGeneratedCallback={this.textGenerated}/>
        </Row>
        <Row className="result-container">
          <p className="result">{this.state.generatedText}</p>
        </Row>
      </Container>
    );
  }
}

export default App;
