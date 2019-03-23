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

class TextGenerator extends React.Component {

  constructor(props) {
    super(props);
    // Create refs to get input
    this.outputLenRef = React.createRef();
    this.seedTextRef = React.createRef();
    this.mlModelRef = React.createRef();
    this.corpusRef = React.createRef();
  }

  handleSubmit(event, outputLen, seedText, mlModel, corpus) {
    event.preventDefault();
    console.log(outputLen)
    console.log(seedText)
    console.log(mlModel)
    console.log(corpus)
  }

  render() {

    return (
        <Form 
          className="text-generator-form" 
          onSubmit = {e => this.handleSubmit(e,
                                             this.outputLenRef.current.value,
                                             this.seedTextRef.current.value,
                                             this.mlModelRef.current.value,
                                             this.corpusRef.current.value)}>
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