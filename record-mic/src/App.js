import React, { useState, useRef } from 'react';
import './App.css';
import { ReactMic } from 'react-mic';

function App() {
  const audioSource = useRef(null);
  const [record, setRecord] = useState(false);
  const [blobURL, setBlobObject] = useState(null);

  const startRecording = () => {
    setRecord(true);
  }

  const stopRecording = (blob) => {
    setRecord(false);
  }

  const onStop = (blob) => {
    setBlobObject(blob);
  }

  const beginCalc = () => {
  }

  const onData = (recordedBlob) => {
    setBlobObject(recordedBlob)
  }

  return (
    <div className="App">
      <header className="App-header">
      <ReactMic
        record={record}         // defaults -> false.  Set to true to begin recording
        className={"sound-wave"}       // provide css class name
        onStart={() => beginCalc()}
        onStop={() => onStop()}        // callback to execute when audio stops recording
        onData={() => onData()}        // callback to execute when chunk of audio data is available
        strokeColor={"#000000"}     // sound wave color
        backgroundColor={"#FF4081"} // background color
        />
      <button onClick={() => startRecording()} type="button">Start</button>
      <button onClick={() => stopRecording()} type="button">Stop</button>
      {record && <p> Recording </p>}
      <div>
        <audio controls="controls" src={blobURL}></audio>
      </div>
      </header>
    </div>
  );
}

export default App;
