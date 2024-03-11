/**
  * Initializes groupStyles object with default color setting each group in the network visualization.
  *
  * @param {groupsArray} array - Array of structure [{"group": "group_id1"}, {"group": "group_id2"}, ...]
  * @returns {groupStyles} object - Object with entries for each group_id containing defaultNodeColor styling.
  */
let generateDefaultGroupList = function (groupsArray, defaultColor){
  // turns groups input from Python into a list of default group stylings
  let groupStyles = {};
  for (groupEntry of groupsArray) {
    groupStyles[groupEntry] = {color: {background: defaultColor}}
  }
  return groupStyles;
}

/** Gets label data for node with specified id.
 * 
 * @param {vis.network} network 
 * @param {string} nodeId 
 * @returns 
 */
function getNodeLabel(network, nodeId){
  // custom function to access network data
  var nodeObj= network.body.data.nodes._data[nodeId];
  let output = nodeObj.label;
  return output;
};

/**
 * 
 * @param {*} edgeList 
 * @param {*} nodeId 
 * @returns 
 */
let filterEdges = function(edgeList, nodeId){
  let filteredEdgeSet = new Set([]);
  if (edgeList.length > 0){
    for (let i = 0; i < edgeList.length; i++) {
      if (edgeList[i].from == nodeId || edgeList[i].to == nodeId ){
        filteredEdgeSet.add(edgeList[i])
      };
    };
    let allEdgesForNode = Array.from(filteredEdgeSet);
    allEdgesForNode.sort((a,b) => a.data.score - b.data.score).reverse();
    let topK = topKSelectionController.getTopKValue();
    let nElemetsToSelect = Math.min(allEdgesForNode.length, topK);
    topKEdgesForNode = allEdgesForNode.slice(0, nElemetsToSelect);
    return topKEdgesForNode;
  };
  return Array.from(filteredEdgeSet)
};

/**
 * 
 * @param {*} inputNodeData 
 * @param {*} selectedNodeId 
 * @returns 
 */
let getNodeStatsInfo = function (inputNodeData, selectedNodeId){
  // function expects selectedNode["data"] information
  let outputString = 'Univariate Data for clicked node with id = ' + String(selectedNodeId) + "\n";
  //console.log("Checking nodeData", inputNodeData)
  for (const contrastKey in inputNodeData) {
    outputString += `[${contrastKey}:]\n`
    for (const measureKey in inputNodeData[contrastKey]){
      if (["globalTestFeaturePValue", "log2FoldChange"].includes(measureKey)){
        // only these two measures have a measure and nodeSize difference in their data.
        value = inputNodeData[contrastKey][measureKey]["measure"];
      } else {
        value = inputNodeData[contrastKey][measureKey];
      }
      outputString += `  ${measureKey}: ${value}\n`
    }
  }
  return outputString
};

/**
 * Extracts node label for specific node from network.
 *
 * @param {network} array - Array of structure [{"group": "group_id1"}, {"group": "group_id2"}, ...]
 * @param {network} array - Array of structure [{"group": "group_id1"}, {"group": "group_id2"}, ...]
 * @returns {string} output - Object with entries for each group_id containing defaultNodeColor styling.
**/
function getNodeGroup(network, nodeId){
  // custom function to access network data
  var nodeObj= network.body.data.nodes._data[nodeId];
  return nodeObj.group;
}

/** Overwrites every group style entry to make use of color.
 * 
 * @param {*} drawingOptions 
 * @param {*} color 
 * @returns 
 */
let resetGroupDrawingOptions = function (drawingOptions, color) {
  for (let [key, value] of Object.entries(drawingOptions["groups"])) {
    drawingOptions["groups"][key]["color"]["background"] = color;
  }
  return undefined // ... drawingOptions modified in place, no return value.
}

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

/**
 * 
 * @param {*} inputGroupData 
 * @param {*} groupId 
 * @returns 
 */
let getNodeGroupInfo = function (inputGroupData, groupId){
  // function expected selectedGroup["data"] information
  let outputString = 'Group-based data for feature-set with id =  ' + String(groupId) + "\n";
  console.log("Checking groupData", inputGroupData)
  for (const contrastKey in inputGroupData) {
    outputString += `[${contrastKey}:]\n`
    for (const measureKey in inputGroupData[contrastKey]){
      rounded_value = inputGroupData[contrastKey][measureKey].toFixed(4)
      outputString += `  ${measureKey}: ${rounded_value}\n`
    }
  }
  return outputString
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



/** Function handles network click response
 * 
 * @param {*} clickInput 
 * @param {*} network 
 * @param {*} networkNodeData 
 * @param {*} networkEdgeData 
 * @param {*} edges 
 * @param {*} groupStats
 * @param {*} networkDrawingOptions
 */
let networkClickController = function(clickInput, network, networkNodeData, networkEdgeData, edges, groupStats, networkDrawingOptions) {
  // If a node is clicked, addon visualization is added,
  // If an empty area is clicked, edge and styling data is reset
  if (clickInput.nodes.length > 0){
    let selectedNode = clickInput.nodes[0]; // assumes only single selections possible!
    let edgeSubset = filterEdges(edges, selectedNode);
    let nodeGroup = getNodeGroup(network, selectedNode);
    let infoString;
    networkEdgeData.update(edgeSubset);
    infoGroupLevel = getNodeGroupInfo(groupStats[nodeGroup], nodeGroup);
    resetGroupDrawingOptions(networkDrawingOptions, stylingVariables.defaultNodeColor);
    highlightTargetGroup(networkDrawingOptions, nodeGroup, stylingVariables.colorHighlight);
    network.storePositions();
    var clickedNode = networkNodeData.get(selectedNode);
    infoString = getNodeStatsInfo(clickedNode["data"], selectedNode);
    nodeInfoContainer.innerText = infoString + infoGroupLevel;
    network.setOptions(networkDrawingOptions);
    network.redraw();
  } else {
    nodeInfoContainer.innerText = "";
    network.storePositions();
    networkEdgeData.clear();
    resetGroupDrawingOptions(networkDrawingOptions, stylingVariables.defaultNodeColor);
    network.setOptions(networkDrawingOptions);
    network.redraw();
  }
};



/** Function handles change in measure or contrast selections
 * 
 * @param {*} networkNodeData 
 * @param {*} network 
 * @returns
 */
let eventHandlerNodeDataChange = function(networkNodeData, network){
  let selectedContrast = formSelectContrast.value;
  let selectedMeasure = formSelectUnivMeasure.value;
  /** Function updates the network data for visualization
   * 
   * @param {*} networkNodeData 
   * @param {*} selectedContrast 
   * @param {*} selectedMeasure 
   * @returns 
   */
  let updateNetworkNodeData = function(networkNodeData, selectedContrast, selectedMeasure) {
    let updatedNodes = [];
    let allNodeIds = networkNodeData.getIds();
    for (var i = 0; i < allNodeIds.length; i++) {
      // For each node, replace the size with the size from the contrast measure selection
      var nodeToUpdate = networkNodeData.get(allNodeIds[i]);
      nodeToUpdate["size"] = nodeToUpdate["data"][selectedContrast][selectedMeasure]["nodeSize"];
      updatedNodes.push(nodeToUpdate);
    };
    networkNodeData.update(updatedNodes);
    return networkNodeData;
  };

  /** Function updates the network view to reflect network data changes
   * 
   * @param {*} network 
   */
  let updateView = function(network){
    network.redraw();
  };
  updateNetworkNodeData(networkNodeData, selectedContrast, selectedMeasure); // in place modification
  updateView(network); // in place modification
  return undefined
}

/** Function resizes network upon window resizing (updates both model and view)
*/
eventHandlerWindowResize = function(network){
  network.fit();
};


/** Function controls response to keydown event on coordinate scaling input. 
 * 
 * @param {*} keydown 
 * @param {*} nodes 
 * @param {*} networkNodeData 
 * @param {*} network 
 */
let eventHandlerCoordinateScaling = function (keydown, nodes, networkNodeData, network, scalingInput){
  // Function only rescales upon enter click
  if (keydown.key === 'Enter') {
    nodes = resizeLayout(scalingInput.value, nodes); // updates the node data
    networkNodeData.clear();
    networkNodeData.update(nodes);
    eventHandlerNodeDataChange(networkNodeData, network);
    network.fit();
  };
};

function initializeInteractiveVisualComponents(nodes, edges, groups, groupStats){
  let networkDrawingOptions;
  let groupList;
  let fullEdgeData; // used for force directed layout only
  let networkNodeData;
  let networkEdgeData;
  let networkData; 
  let network;
  
  groupList = generateDefaultGroupList(groups, stylingVariables.defaultNodeColor);
  networkDrawingOptions = generateNetworkDrawingOptions(groupList);
  // Add node title information based on node id and node group:
  nodes.forEach(function(node) {
    node.title = 'ID: ' + node.id + '<br>Group: ' + node.group;
  });
  // Processing Edges; rounding labels to make sure they are printable on edges
  for (let edge of edges){
    edge["label"] = `${edge["data"]["score"].toFixed(2)}`;
  };
  edges.forEach(function(edge) {
    edge.title = 'ID: ' + edge.id + '<br>Score: ' + edge.data.score;
  });

  fullEdgeData = new vis.DataSet(edges);
  networkNodeData = new vis.DataSet(nodes);
  networkEdgeData = new vis.DataSet([]);

  // construct network variable and attach to div
  networkData = {nodes: networkNodeData, edges: networkEdgeData};
  network = new vis.Network(networkContainer, networkData, networkDrawingOptions);
  heatmapPanelController(groupStats, formSelectContrast, networkDrawingOptions, network);
  // Init Run NodeChangeData handler to ensure match between selected options and display data
  eventHandlerNodeDataChange(networkNodeData, network);

  // Define Network Callback Events & Responses
  network.on("dragging", () => function(network){network.storePositions();});
  
  /*
  network.on("dragging", function (params){
    network.storePositions();
    return undefined;
  })
  */

  network.on(
    "click", 
    input => networkClickController(
      input, network, networkNodeData, networkEdgeData, edges, groupStats, networkDrawingOptions
    )
  );

  runNetworkPhysicsInput.addEventListener(
    'keydown', 
    keyInput => networkStabilizationController(
      keyInput, network, networkEdgeData, fullEdgeData
    )
  );

  formSelectContrast.addEventListener(
    "change",
    () => eventHandlerNodeDataChange(networkNodeData, network)
  );

  formSelectUnivMeasure.addEventListener(
    "change", 
    () => eventHandlerNodeDataChange(networkNodeData, network)
  );

  scalingInput.addEventListener(
    'keydown', 
    (keydown) => eventHandlerCoordinateScaling(keydown, nodes, networkNodeData, network, scalingInput)
  );
  window.onresize = eventHandlerWindowResize(network)
};

buttonLoadJsonData.addEventListener("click", dataLoadingController);



