import * as tf from '@tensorflow/tfjs'

function face_mesh_test(endpoint){
    const model = loadModel(endpoint)

    async function loadModel(endpoint) {
        try {
          const model = await tf.loadGraphModel(endpoint);
          return model;
    
        }
        catch (err) {
          console.log(err);
          console.log("failed load model");
    
      }}

      function predict(image){
        let tensorImg =   tf.browser.fromPixels(image).resizeNearestNeighbor([192, 192]).toFloat().expandDims(0).div(255.0)
        console.log(tensorImg)
        const predictions = await model.predict(tensorImg);
        return predictions
      }
}

module.exports.face_mesh_test = face_mesh_test;