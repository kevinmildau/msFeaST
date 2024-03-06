// DOM References
let networkContainer = document.getElementById("id-visjs-network-example");
let nodeInfoContainer = document.getElementById("id-node-information");
let svgExportButton = document.getElementById("id-save-to-svg");
let jsonPrintContainer = document.getElementById("id-json-print-container");
let buttonLoadJsonData = document.getElementById("id-button-load-json");
let fromTopkSelector = document.getElementById("id-topk-range-input");
let formSelectContrast = document.getElementById("id-contrast-selector" );
let formSelectUnivMeasure = document.getElementById("id-univ-measure-selector");
let topkLivePrintContainer = document.getElementById("id-topk-live-print");
let heatmapContainer = document.getElementById("id_heatmap");
let runNetworkPhysicsInput = document.getElementById("id-network-physics");
let colorBarContainer = document.getElementById('id_color_bar');

function initializeInteractiveVisualComponents(nodes, edges, groups, groupStats, domElementContrast, domElementMeasure) {
  // Constants
  const colorHighlight = "rgb(235,16,162, 0.9)"// "#EB10A2"
  const defaultEdgeColor = "#A65A8D"
  const defaultNodeColor = "rgb(166, 90, 141, 0.5)" // "#A65A8D"
  const defaultNodeBorderColor = "rgb(0, 0, 0, 0.5)"

  /**
   * Updates borderWidth of selected node to double size. For use inside vis.js network object.
   *
   * @param {values} object - visjs network selected return with styling elements accessible.
  **/
  const chosenHighlightNode = function(values){
    values.borderWidth = values.borderWidth * 2;
  }

  /**
   * Updates borderWidth of selected node to double size. For use inside vis.js network object.
   *
   * @param {values} object - visjs network selected return with styling elements accessible.
  **/
  const chosenHighlightEdge = function(values){
    values.color = colorHighlight;
    values.opacity = 0.9;
  }

  /**
   * Initializes groupStyles object with default color setting each group in the network visualization.
   *
   * @param {groupsArray} array - Array of structure [{"group": "group_id1"}, {"group": "group_id2"}, ...]
   * @returns {groupStyles} object - Object with entries for each group_id containing defaultNodeColor styling.
  **/
  let generateDefaultGroupList = function (groupsArray, defaultColor){
    // turns groups input from Python into a list of default group stylings
    let groupStyles = {};
    for (groupEntry of groupsArray) {
      groupStyles[groupEntry] = {color: {background: defaultColor}}
    }
    return groupStyles
  }
  
  let groupList = generateDefaultGroupList(groups, defaultNodeColor)
  
  // This structure contains any styling used for the network.
  // This structure is modified to recolor groups if a node belonging to the group is selected.
  // Here, the groups color attribute is changed. 
  const networkDrawingOptions = {
    physics: false,
    nodes: {
      shape: "dot", // use dot, circle scales to the label size as the label is inside the shape! 
      chosen: {node: chosenHighlightNode}, // this passes the function to style the respective selected node
      color: {background: defaultNodeColor, border: defaultNodeBorderColor},
      size: 25, font: {size: 14, face: "Helvetica"}, borderWidth: 1, 
    },
    edges: {
      chosen: {edge:chosenHighlightEdge}, // this passes the function to style the respective selected edge
      font:  {size: 14, face: "Helvetica"},
      color: { opacity: 0.6, color: defaultEdgeColor, inherit: false},
      smooth: {type: "straightCross", forceDirection: "none", roundness: 0.25},
    },
    interaction: {
      selectConnectedEdges: true, // prevent edge highlighting upon node selection
      hover:true, hideEdgesOnDrag: false, tooltipDelay: 100
    },
    // On data load, add this group-information such that all groups are set to default node color
    // when a node is clicked, node group can be accessed to change color and redraw.
    // It is not possible to override the default coloring of groups any other way than specifying a replacement default 
    // color here. Any other solution would involve direct node modification (change color of each specific node).
    groups: groupList,
  }

  let resizeLayout = function (scalingValue){
    // nodeData: array of objects with x and y numeric data
    // function modified nodes object directly
    if (scalingValue > 0){
      const xValues = nodes.map(obj => obj.x);
      const yValues = nodes.map(obj => obj.y);
      const xMinValue = Math.min(...xValues);
      const xMaxValue = Math.max(...xValues);
      const yMinValue = Math.min(...yValues);
      const yMaxValue = Math.max(...yValues);

      if (xMinValue === xMaxValue){console.error("Error: xmin and xmax coordinates are identical!")}
      if (yMinValue === yMaxValue){console.error("Error: ymin and ymax coordinates are identical!")}

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
  }
  // Add node title information based on node id and node group:
  nodes.forEach(function(node) {
    node.title = 'ID: ' + node.id + '<br>Group: ' + node.group;
  });
  edges.forEach(function(edge) {
    edge.title = 'ID: ' + edge.id + '<br>Score: ' + edge.data.score;
  });

  let fullEdgeData = new vis.DataSet(edges);
  let networkNodeData = new vis.DataSet();
  let networkEdgeData = new vis.DataSet();
  networkNodeData.add(nodes)
  networkEdgeData.add([])
  let networkData = {nodes: networkNodeData, edges: networkEdgeData}
  console.log(networkNodeData)

  // Network data extraction functions

    /**
   * Extracts node label for specific node from network.
   *
   * @param {network} array - Array of structure [{"group": "group_id1"}, {"group": "group_id2"}, ...]
   * @param {network} array - Array of structure [{"group": "group_id1"}, {"group": "group_id2"}, ...]
   * @returns {string} output - Object with entries for each group_id containing defaultNodeColor styling.
  **/
  function getNodeLabel(network, nodeId){
    // custom function to access network data
    var nodeObj= network.body.data.nodes._data[nodeId];
    let output = nodeObj.label;
    return output;
  }
  function getNodeGroup(network, nodeId){
    // custom function to access network data
    var nodeObj= network.body.data.nodes._data[nodeId];
    return nodeObj.group;
  }
  let resetGroupDrawingOptions = function (drawingOptions, color) {
    /*
    Overwrites every group style entry to make use of color.
    */
    for (let [key, value] of Object.entries(drawingOptions["groups"])) {
      drawingOptions["groups"][key]["color"]["background"] = color;
    }
    return undefined // ... drawingOptions modified in place, no return value.
  }
  let highlightTargetGroup = function (drawingOptions, group, color){
    /*
    Overwrites target group style entry to make use of color.
    */
    drawingOptions["groups"][group]["color"]["background"] = color;
    return undefined // ... drawingOptions modified in place, no return value.
  }
  // construct network variable and attach to div
  network = new vis.Network(networkContainer, networkData, networkDrawingOptions);

  // Define network Event Targets, Event Listeners, and Event Handlers:
  network.on("dragging", function (params){
    network.storePositions();
    return undefined
  })
  network.on("click", function (params) {
    let filterEdges = function(edgeList, nodeId){
      let filteredEdgeSet = new Set([])
      if (edgeList.length > 0){
        for (let i = 0; i < edgeList.length; i++) {
          if (edgeList[i].from == nodeId || edgeList[i].to == nodeId ){
            filteredEdgeSet.add(edgeList[i])
          }
        }
        let allEdgesForNode = Array.from(filteredEdgeSet)
        allEdgesForNode.sort((a,b) => a.data.score - b.data.score).reverse();
        let topK = fromTopkSelector.value;
        let nElemetsToSelect = Math.min(allEdgesForNode.length, topK);
        topKEdgesForNode = allEdgesForNode.slice(0, nElemetsToSelect)
        return topKEdgesForNode
      }
      return Array.from(filteredEdgeSet)
    }
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
    }
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
    if (params.nodes.length > 0){
      //params.event = "[original event]";
      let selectedNode = params.nodes[0] // assumes only single selections possible!

      let edgeSubset = filterEdges(edges, selectedNode)
      networkEdgeData.update(edgeSubset)
      let nodeGroup = getNodeGroup(network, selectedNode)
      let infoString;
      infoGroupLevel = getNodeGroupInfo(groupStats[nodeGroup], nodeGroup)
      resetGroupDrawingOptions(networkDrawingOptions, defaultNodeColor);
      highlightTargetGroup(networkDrawingOptions, nodeGroup, colorHighlight)
      network.storePositions();
      var clickedNode = networkNodeData.get(selectedNode);
      infoString = getNodeStatsInfo(clickedNode["data"], selectedNode)
      nodeInfoContainer.innerText = infoString + infoGroupLevel;
      network.setOptions(networkDrawingOptions);
      network.redraw();
    } else {
      nodeInfoContainer.innerText = "";
      network.storePositions();
      networkEdgeData.clear();
      resetGroupDrawingOptions(networkDrawingOptions, defaultNodeColor);
      network.setOptions(networkDrawingOptions);
      network.redraw();
    }
  });
  
  runNetworkPhysicsInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      // code for enter
      networkEdgeData.update(fullEdgeData);
      let n_iterations = Number(runNetworkPhysicsInput.value)
      network.stabilize(n_iterations)
      networkEdgeData.clear()
      network.storePositions();
      network.redraw()
      network.fit();
    }
  });

  let adjustNodeDataToSelection = function (){
    console.log("Reached Adjusting Node Size")
    console.log(networkNodeData)
    let selectedContrast = domElementContrast.value;
    let selectedMeasure = domElementMeasure.value;
    // Get all node ids
    var allNodeIds = networkNodeData.getIds();
    var updatedNodes = []
    for (var i = 0; i < allNodeIds.length; i++){
      // For each node, replace the size with the size from the contrast measure selection
      var nodeToUpdate = networkNodeData.get(allNodeIds[i]);
      nodeToUpdate["size"] = nodeToUpdate["data"][selectedContrast][selectedMeasure]["nodeSize"]
      updatedNodes.push(nodeToUpdate)
    }
    networkNodeData.update(updatedNodes)
    network.redraw()
  }
  formSelectContrast.addEventListener("change", adjustNodeDataToSelection);
  formSelectUnivMeasure.addEventListener("change", adjustNodeDataToSelection);
  adjustNodeDataToSelection(); // run on init to make sure size properties of nodes align with form element.
  
  let scalingInput = document.getElementById("id-graph-scaler")
  scalingInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      // code for enter
      resizeLayout(scalingInput.value) // updates the node data
      networkNodeData.clear()
      networkNodeData.update(nodes)
      adjustNodeDataToSelection()
      network.fit();
    }
  });
  
  let constructHeatmapPanel = function(groupStats){
    // Constructs heatmap with global test set specific results
    // Dev Note: Assumes n_measures at set level is equal to 1!
    console.log(groupStats)
    var z = [];
    var ticks = [];

    let nRows = Array.from(domElementContrast.options).map(option => option.value).length; // number of contrasts
    let nCols = Object.keys(groupStats).length; // number of groups
    let yTicks = Array.from(domElementContrast.options).map(option => option.value);
    let xTicks = Object.keys(groupStats);
    console.log(yTicks)
    console.log(xTicks)
    let values_array = [];
    for (let current_contrast of yTicks) {
      tmp_array = []
      for (let current_group of xTicks) {
        
        tmp_array.push(groupStats[current_group][current_contrast]["globalTestPValue"])
      }
      values_array.push(tmp_array)
    }
    let colorscale = 'YlGnBu'
    var data = [{
      z: values_array,
      x: xTicks,
      y: yTicks,
      zmin: 0,  // Minimum color value
      zmax: 1,   // Maximum color value
      colorscale: colorscale, // [[0, 'white'], [1, 'blue']], // for custom color scale
      showscale : false,
      //reversescale : true,
      type: 'heatmap',
      hovertemplate: 'x: %{x}<br>y: %{y}<br>z: %{z:.2e}<extra></extra>'
    }];

    let scale_values = Array.from({length: 101}, (_, i) => i * 0.01);
    console.log(scale_values)
    var colorBarHeatmap = [{
      z: [scale_values], // This should match the color levels of your original heatmap
      x: scale_values,
      y: ["P-value color gradient"],
      type: 'heatmap',
      colorscale: colorscale,
      showscale : false,
      xaxis: 'p-value',
      yaxis: ['color'],
      hovertemplate: 'p-value: %{x}<br><extra></extra>',
    }];
    let margin = 10;
    var layout_colorbar = {
      title : "",
      margin: {l: margin,r: margin,b: margin, t: margin,}, 
      xaxis: {
        automargin : true,
        domain: [0, 1],
        tickfont : {size : 8},
        anchor: 'y',
      },
      yaxis: {
        anchor: 'x',
        automargin : true,
        showticklabels: true,
        side: "right",
        tickmode: "linear", 
        dtick:1
      },
      autosize: true, // Automatically adjust size to container
    };
    Plotly.newPlot(colorBarContainer, colorBarHeatmap, layout_colorbar, {responsive : true});

    var layout = {
      //margin: {l: margin,r: margin,b: margin,t: margin,}, 
      margin: {t:margin, l:margin},
      xaxis: {automargin : true, tickmode: "linear", dtick:1, tickfont : {size : 8}},
      yaxis: {automargin : true, side: "right"},
      //xaxis : {tickmode:'array', tickvals:ticks, ticktext:ticks},
      //yaxis : {fixedrange : true, tickmode: 'array', tickvals: ticks, ticktext: ticks},
      title: '',
      autosize: true, // Automatically adjust size to container
    };
    Plotly.newPlot(heatmapContainer, data, layout, {responsive : true});
    var hoverTimer;  // Define a timer variable
    /*
    Delayed Plotly hover response code.
    */
    var delay_hover = 500; // Delay in milliseconds
    heatmapContainer.on('plotly_hover', function(data){
      // Clear the timer if it's already set
      if(hoverTimer) {
        clearTimeout(hoverTimer);
      }
      // Set a new timer
      hoverTimer = setTimeout(function() {
        var xValue = data.points[0].x;
        //document.getElementById("textout").innerText = "Last hovered over y value: " + yValue;
        console.log('Hovering over x value: ' + xValue);
        resetGroupDrawingOptions(networkDrawingOptions, defaultNodeColor) // autoreset at every hover
        highlightTargetGroup(networkDrawingOptions, xValue, colorHighlight)
        network.setOptions(networkDrawingOptions);
        network.redraw();
      }, delay_hover);
    });
    /*
    var delay_click = 10; // Delay in milliseconds
    heatmapContainer.on('plotly_click', function(data){
      // Clear the timer on unhover
      if(hoverTimer) {
        clearTimeout(hoverTimer);
      }
      hoverTimer = setTimeout(function() {
        //document.getElementById("textout").innerText = "Hover cleared.";
        resetGroupDrawingOptions(networkDrawingOptions, defaultNodeColor)
        network.setOptions(networkDrawingOptions);
        network.redraw();
      }, delay_click);
    });
    */
    
  }
 
  constructHeatmapPanel(groupStats)

  window.onresize = function() {network.fit();}
}

function loadDataAndConstructNetworkVisualization() {
  let input, file, fr;
  let receivedText = function (e) {
    let lines = e.target.result;
    let inputData;
    try {
      inputData = JSON.parse(lines);
    } catch (error) {
      alert(`Input data does not appear to conform with JSON file standard. Aborting data loading. Following error was generated: ${error}`)
      return undefined
    }

    let validateInputSchema = function (inputData){
      // basic function to check whether the expected entries are available, to be replaced with object schema and type validation
      // inputData is expected to be type object after a successful json parse. JSON parsing errors are handled separately.
      let problemDetected = false; 
      let expectedKeys = ["groupKeys", "univMeasureKeys", "groupMeasureKeys", "contrastKeys", "groupStats", "nodes", "edges"]
      for (key of expectedKeys){
        if (typeof(inputData[key]) === "undefined"){
          problemDetected = true; 
          // The console log below is part of the desired console output
          console.log(`Key entry "${key}" not found in input data.`);
        }
      }
      return problemDetected
    }

    if (validateInputSchema(inputData)){
      alert("Provided file does not contain expected input structues. See console log for missing entry. Aborting file loading and interactive session initialization.");
      return undefined
    }

    let nodes = inputData["nodes"];
    let edges = inputData["edges"];
    let groupKeys = inputData["groupKeys"];
    let univMeasureKeys = inputData["univMeasureKeys"];
    let groupMeasureKeys = inputData["groupMeasureKeys"];
    let contrastKeys = inputData["contrastKeys"];
    let groupStats = inputData["groupStats"]

    // Processing Edges; rounding labels to make sure they are printable on edges
    for (let edge of edges){
      edge["label"] = `${edge["data"]["score"].toFixed(2)}`
    } 

    let initializeContrastMeasureSelectors = function(domElementContrast, domElementMeasure, optionsArrayContrastKeys, optionsArrayMeasureKeys, nodes, groupKeys, groupStats) {
      domElementContrast.innerHTML = "";
      domElementMeasure.innerHTML = "";
      optionsArrayContrastKeys.forEach(function(optionKey) {
        var option = document.createElement("option");
        option.value = optionKey;
        option.text = optionKey;
        domElementContrast.appendChild(option);
      });
      optionsArrayMeasureKeys.forEach(function(optionKey) {
        var option = document.createElement("option");
        option.value = optionKey;
        option.text = optionKey;
        domElementMeasure.appendChild(option);
      });
      initializeInteractiveVisualComponents(nodes, edges, groupKeys, groupStats, domElementContrast, domElementMeasure)
    }
    
    initializeContrastMeasureSelectors(formSelectContrast, formSelectUnivMeasure, contrastKeys, univMeasureKeys, nodes, groupKeys, groupStats)
  }
  if (typeof window.FileReader !== 'function') {
    alert("The file API isn't supported on this browser yet.");
    return;
  }
  input = document.getElementById('id-file-input');
  if (!input) {
    alert("File input dom element not found!");
  }
  else if (!input.files) {
    alert("This browser does not seem to support the `files` property of file inputs.");
  }
  else if (!input.files[0]) {
    alert("Please select a file before clicking 'Load'");
  }
  else {
    file = input.files[0];
    fr = new FileReader();
    fr.addEventListener("load", receivedText);
    //fr.onload = receivedText;
    fr.readAsText(file);
  }
}
buttonLoadJsonData.addEventListener("click", loadDataAndConstructNetworkVisualization)


// Bootstrap component nav tab change listener to force window-resize event upon chaning tabs
// this is needed for plotly.js and vis.js to properly size the visuals inside the tabs.
let navTabs = document.querySelector('.nav-pills');

// Add a listener function to the shown.bs.tab event
navTabs.addEventListener('shown.bs.tab', function (e) {
  // e.target is the new active tab
  // e.relatedTarget is the previous active tab

  // Trigger the window resize event
  window.dispatchEvent(new Event('resize'));
});

let displaytopkSelected = function (){
  topkLivePrintContainer.innerHTML = `Selected top-K value: ${this.value}`;
}
fromTopkSelector.addEventListener("change", displaytopkSelected);