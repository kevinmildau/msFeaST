/** Function rescales x y coordinates to fit between [-1, +1] times scalingValue. Linear scaling.
 * 
 * @param {*} scalingValue A non-zero float
 * @param {*} nodes A list of node entries for vis network
 * @returns 
 */
let resizeLayout = function (scalingValue, nodes){
  // nodeData: array of objects with x and y numeric data
  // function modified nodes object directly
  if (scalingValue > 0){
    const xValues = nodes.map(obj => obj.x);
    const yValues = nodes.map(obj => obj.y);
    const xMinValue = Math.min(...xValues);
    const xMaxValue = Math.max(...xValues);
    const yMinValue = Math.min(...yValues);
    const yMaxValue = Math.max(...yValues);

    if (xMinValue === xMaxValue){console.error("Error: xmin and xmax coordinates are identical!")};
    if (yMinValue === yMaxValue){console.error("Error: ymin and ymax coordinates are identical!")};

    const newMax = 1 * scalingValue;
    const newMin = -1 * scalingValue;
    const xScaleFactor = (newMax - newMin) / (xMaxValue - xMinValue);
    const ySaleFactor = (newMax - newMin) / (yMaxValue - yMinValue);
    // Assess current scale of x and y values
    // Set scale to unit scale and multiply with scalingValue
    for (let node of nodes){
      node.x = (node.x - xMinValue) * xScaleFactor + newMin; 
      node.y = (node.y - yMinValue) * ySaleFactor + newMin;
    }
  }
  return nodes;
}