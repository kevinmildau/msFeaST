// Constants and helpers
const margin = 25;
const layout = {
  margin: {l: margin,r: margin,b: margin,t: margin,}, 
  font: {family: "Helvetica", size: 10}
};
const config = {responsive: true}

let $PlotlyExample = document.getElementById("id_plotly_example")

let trace1 = {
  x: [1, 2, 3, 4],
  y: [10, 15, 13, 17],
  mode: "markers",
  type: "scatter"
};
let trace2 = {
  x: [2, 3, 4, 5],
  y: [16, 5, 11, 9],
  mode: "lines",
  type: "scatter"
};
let trace3 = {
  x: [1, 2, 3, 4],
  y: [12, 9, 15, 12],
  mode: "lines+markers",
  type: "scatter"
};


let data = [trace1, trace2, trace3];
// newPlot takes as first argument the dom node or the identifier of the dom node as a string
Plotly.newPlot($PlotlyExample,  data, layout, config);
