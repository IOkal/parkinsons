//-------- inference the neural network --------//

const readUploadedFileAsText = inputFile => {
  const temporaryFileReader = new FileReader();

  return new Promise((resolve, reject) => {
    temporaryFileReader.onerror = () => {
      temporaryFileReader.abort();
      reject(new DOMException("Problem parsing input file."));
    };

    temporaryFileReader.onload = () => {
      console.log(temporaryFileReader);
      resolve(temporaryFileReader.result);
    };
    temporaryFileReader.readAsBinaryString(inputFile);
  });
};

async function inference(file) {
  console.log(file);
  console.log("Entering inference function");
  var data = await readUploadedFileAsText(file);
  console.log(data);
  var response = await lib.tobcar["parkinsons-classification"]["@dev"]({
    file: data
  });

  //document.getElementById('prediction').value = response.prediction
  //document.getElementById('averageFundamentalFrequency').value = response.averageFundamentalFrequency
  //document.getElementById('jitter').value = response.jitter
  //document.getElementById('shimmer').value = response.shimmer

  console.log(response);
}

function handleFiles(files) {
  files = [...files];
  inference(files[0]);
}
