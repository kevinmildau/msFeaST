
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

/** Overwrites every group style entry to make use of color.
 * 
 * @param {*} drawingOptions 
 * @param {*} color 
 * @returns 
 */
const resetGroupDrawingOptions = function (drawingOptions, color) {
  for (let [key, value] of Object.entries(drawingOptions["groups"])) {
    drawingOptions["groups"][key]["color"]["background"] = color;
  }
  return undefined // ... drawingOptions modified in place, no return value.
}

/** Generates default network drawing options.
 * 
 * @param {*} groupList 
 * @returns 
 */
const generateNetworkDrawingOptions = function(groupList){
  // This structure contains any styling used for the network.
  // This structure is modified to recolor groups if a node belonging to the group is selected.
  // Here, the groups color attribute is changed. 
  // On data load, add this group-information such that all groups are set to default node color.
  // When a node is clicked, node group can be accessed to change color and redraw.
  // It is not possible to override the default coloring of groups any other way than specifying a replacement 
  // color here. Any other solution would involve direct node modification (change color of each specific node).
  let networkDrawingOptions;
  networkDrawingOptions = {
    groups: groupList,
    physics: false,
    nodes: {
      shape: "dot", // use dot, circle scales to the label size as the label is inside the shape! 
      chosen: {node: stylizeHighlightNode}, // this passes the function to style the respective selected node
      color: {background: stylingVariables.defaultNodeColor, border: stylingVariables.defaultNodeBorderColor},
      size: 25, font: {size: 14, face: "Helvetica"}, borderWidth: 1, 
    },
    edges: {
      chosen: {edge:stylizeHighlightEdge}, // this passes the function to style the respective selected edge
      font:  {size: 14, face: "Helvetica"},
      color: { opacity: 0.6, color: stylingVariables.defaultEdgeColor, inherit: false},
      smooth: {type: "straightCross", forceDirection: "none", roundness: 0.25},
    },
    interaction: {
      selectConnectedEdges: true, // prevent edge highlighting upon node selection
      hover:true, hideEdgesOnDrag: false, tooltipDelay: 100
    },

  };
  return networkDrawingOptions;
};

/** Overwrites target group style entry to make use of color.
 * 
 * @param {*} drawingOptions 
 * @param {*} group 
 * @param {*} color 
 * @returns 
 */
let highlightTargetGroup = function (drawingOptions, group, color){
  drawingOptions["groups"][group]["color"]["background"] = color;
  return undefined // ... drawingOptions modified in place, no return value.
}