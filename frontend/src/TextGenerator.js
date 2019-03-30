import React from 'react';

import Form from 'react-bootstrap/Form'
import Col from 'react-bootstrap/Col'
import Row from 'react-bootstrap/Row'
import Button from 'react-bootstrap/Button'

// Constants for form fields
const MODEL_LSTM = "LSTM"
const MODEL_MARKOV = "Markov Chain"
const CORPUS_SHAKESPEARE = "Shakespeare"
const CORPUS_TRUMP = "Trump Tweets"
const BASE_URL = "whatisthis??" // TODO fill me out! 

class TextGenerator extends React.Component {

  constructor(props) {
    super(props);
    // Create refs to get input
    this.outputLenRef = React.createRef();
    this.seedTextRef = React.createRef();
    this.mlModelRef = React.createRef();
    this.corpusRef = React.createRef();
  }

  handleSubmit(event, outputLen, seedText, mlModel, corpus, callback) {
    event.preventDefault();
    // For debugging
    console.log(outputLen)
    console.log(seedText)
    console.log(mlModel)
    console.log(corpus)
    // Decide which endpoint to hit, set some defaults
    let model = "lstm"
    if (mlModel === MODEL_MARKOV) {
      model = 'markov'
    }
    let corpus = "trump"
    if (corpus === CORPUS_SHAKESPEARE) {
      corpus = "shakespeare"
    }
    // Build request body
    let reqData = {
      "string_length": outputLen, // TODO could probably convert into a number here, but it is done in backend
      "seed_text": seedText
    }
    // Build URL
    let url = BASE_URL + "/model/" + model + "/" + corpus
    // Send request
    fetch(url, {
      method: "GET",
      body: JSON.stringify(reqData),
      headers: {
        "Content-Type": "application/json"
      }
    }).then(function(response) {
      // Debugging
      console.log(response.status)
      console.log(response.text())
      // Call the callback with the text generated
      callback(response.text())
    }, function(error) {
      console.log("ERROR: " + error.message)
    })
  }

  render() {

    return (
        <Form 
          className="text-generator-form" 
          onSubmit = {e => this.handleSubmit(e,
                                             this.outputLenRef.current.value,
                                             this.seedTextRef.current.value,
                                             this.mlModelRef.current.value,
                                             this.corpusRef.current.value,
                                             this.props.textGeneratedCallback)}>
        <Row>
          <Col>
            <Form.Group controlId="formOutputLen">
              <Form.Label>Output Length</Form.Label>
              <Form.Control type="text" placeholder="ex. 42" ref={this.outputLenRef} required/>
            </Form.Group>
          </Col>
          <Col>
            <Form.Group controlId="formSeedText">
                <Form.Label>Seed Text</Form.Label>
                <Form.Control type="text" placeholder="Where art thou..." ref={this.seedTextRef} required/>
            </Form.Group>
          </Col>
        </Row>
        <Row>
          <Col>
          <Form.Group controlId="formModel">
            <Form.Label>ML Model</Form.Label>
            <Form.Control
              as="select"
              ref={this.mlModelRef}>
              <option>{MODEL_LSTM}</option>
              <option>{MODEL_MARKOV}</option>
            </Form.Control>
          </Form.Group>
          </Col>
          <Col>
          <Form.Group controlId="formCorpus">
            <Form.Label>Corpus</Form.Label>
            <Form.Control 
              as="select"
              ref={this.corpusRef}>
              <option>{CORPUS_SHAKESPEARE}</option>
              <option>{CORPUS_TRUMP}</option>
            </Form.Control>
          </Form.Group>
          </Col>
        </Row>
        <Button variant="primary" type="submit">
          Submit
        </Button>
      </Form>
    );
  }
}

export default TextGenerator;