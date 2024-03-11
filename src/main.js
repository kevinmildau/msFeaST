/** Function handles network drag event.
 * 
 * @param {*} dragNodeData 
 * @param {*} network 
 */
let networkDragController = function(dragNodeData, network){
  network.storePositions(); 
}

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
    let clickedNode
    networkEdgeData.update(edgeSubset);
    infoGroupLevel = getNodeGroupInfo(groupStats[nodeGroup], nodeGroup);
    resetGroupDrawingOptions(networkDrawingOptions, stylingVariables.defaultNodeColor);
    highlightTargetGroup(networkDrawingOptions, nodeGroup, stylingVariables.colorHighlight);
    network.storePositions();
    clickedNode = networkNodeData.get(selectedNode);
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

/** Function handles change in measure or contrast selections
 * 
 * @param {*} networkNodeData 
 * @param {*} network
 * @returns
 */
let updateNodeDataToContrastAndMeasure = function(networkNodeData, network){
  let selectedContrast = formSelectContrast.value;
  let selectedMeasure = formSelectUnivMeasure.value;
  updateNetworkNodeData(networkNodeData, selectedContrast, selectedMeasure); // in place modification
  network.redraw();
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
let eventHandlerCoordinateScaling = function (keydown, networkNodeData, network, scalingInput){
  // Function only rescales upon enter click
  if (keydown.key === 'Enter') {
    updatedNodes = resizeLayout(scalingInput.value, networkNodeData); // updates the node data
    networkNodeData.clear();
    networkNodeData.update(updatedNodes);
    //updateNodeDataToContrastAndMeasure(networkNodeData, network);
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

  // Initialize Visual Interaction Event Controllers
  heatmapPanelController(groupStats, formSelectContrast, networkDrawingOptions, network);
  // Init Run NodeChangeData handler to ensure match between selected options and display data
  updateNodeDataToContrastAndMeasure(networkNodeData, network);

  network.on("dragging", (dragNodeData) => networkDragController(dragNodeData, network));

  network.on("click", input => networkClickController(
    input, network, networkNodeData, networkEdgeData, edges, groupStats, networkDrawingOptions
    )
  );

  runNetworkPhysicsInput.addEventListener('keydown', 
    keyInput => networkStabilizationController(
      keyInput, network, networkEdgeData, fullEdgeData
    )
  );

  formSelectContrast.addEventListener(
    "change",
    () => updateNodeDataToContrastAndMeasure(networkNodeData, network)
  );

  formSelectUnivMeasure.addEventListener(
    "change", 
    () => updateNodeDataToContrastAndMeasure(networkNodeData, network)
  );

  scalingInput.addEventListener(
    'keydown', 
    (keydown) => eventHandlerCoordinateScaling(keydown, networkNodeData, network, scalingInput)
  );
  window.onresize = eventHandlerWindowResize(network)
};

buttonLoadJsonData.addEventListener("click", dataLoadingController);



