
// DOM References
const networkContainer = document.getElementById("id-visjs-network-example");
const nodeInfoContainer = document.getElementById("id-node-information");
const jsonPrintContainer = document.getElementById("id-json-print-container");
const buttonLoadJsonData = document.getElementById("id-button-load-json");
const fromTopkSelector = document.getElementById("id-topk-range-input");
const topkLivePrintContainer = document.getElementById("id-topk-live-print");
const formSelectContrast = document.getElementById("id-contrast-selector" );
const formSelectUnivMeasure = document.getElementById("id-univ-measure-selector");
const heatmapContainer = document.getElementById("id_heatmap");
const runNetworkPhysicsInput = document.getElementById("id-network-physics");
const colorBarContainer = document.getElementById('id_color_bar');
const scalingInput = document.getElementById("id-graph-scaler");

// Styling Variables
const stylingVariables = {
  colorHighlight: "rgb(235,16,162, 0.9)",
  defaultEdgeColor: "#A65A8D",
  defaultNodeColor: "rgb(166, 90, 141, 0.5)", 
  defaultNodeBorderColor: "rgb(0, 0, 0, 0.5)"
};