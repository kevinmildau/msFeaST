
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

/**
  * Updates borderWidth of selected node to double size. For use inside vis.js network object.
  *
  * @param {values} object - visjs network selected return with styling elements accessible.
  */
const stylizeHighlightNode = function(values){
  values.borderWidth = values.borderWidth * 2;
};

/**
  * Updates borderWidth of selected node to double size. For use inside vis.js network object.
  *
  * @param {values} object - visjs network selected return with styling elements accessible.
  */
const stylizeHighlightEdge = function(values){
  values.color = stylingVariables.colorHighlight;
  values.opacity = 0.9;
};

/**
  * Initializes groupStyles object with default color setting each group in the network visualization.
  *
  * @param {array} groupsArray - Array of structure [{"group": "group_id1"}, {"group": "group_id2"}, ...]
  * @returns {object} groupStyles - Object with entries for each group_id containing defaultNodeColor styling.
  */
const generateDefaultGroupList = function (groupsArray, defaultColor){
  let groupStyles = {};
  for (groupEntry of groupsArray) {
    groupStyles[groupEntry] = {color: {background: defaultColor}}
  }
  return groupStyles;
}