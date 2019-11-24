//const axios = require('axios');

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
  console.log(response);
  // axios
  //   .post(
  //     "https://TobCar.api.stdlib.com/parkinsons-classification@0.1.0/predict/",
  //     {
  //       file: r.readAsBinaryString(file)
  //     },
  //     {
  //       headers: {
  //         "Content-Type": "application/json",
  //         "Access-Control-Allow-Origin": "*"
  //       }
  //     }
  //   )
  //   .then(function(response) {
  //     console.log(response);
  //   })
  //   .catch(function(error) {
  //     console.log(error);
  //   });
}

function handleFiles(files) {
  files = [...files];
  /*initializeProgress(files.length)
  files.forEach(uploadFile)
  files.forEach(previewFile)*/
  inference(files[0]);
}

(function($) {
  "use strict";

  $(document).ready(function() {
    $("select").niceSelect();
  });
  // menu fixed js code
  $(window).scroll(function() {
    var window_top = $(window).scrollTop() + 1;
    if (window_top > 50) {
      $(".main_menu").addClass("menu_fixed animated fadeInDown");
    } else {
      $(".main_menu").removeClass("menu_fixed animated fadeInDown");
    }
  });

  $(document).ready(function() {
    $("select").niceSelect();
  });

  var review = $(".client_review_part");
  if (review.length) {
    review.owlCarousel({
      items: 1,
      loop: true,
      dots: true,
      autoplay: true,
      autoplayHoverPause: true,
      autoplayTimeout: 5000,
      nav: false,
      smartSpeed: 2000
    });
  }

  //------- Mailchimp js --------//
  function mailChimp() {
    $("#mc_embed_signup")
      .find("form")
      .ajaxChimp();
  }
  mailChimp();
})(jQuery);
