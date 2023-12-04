// DOM References
let networkContainer = document.getElementById("id-visjs-network-example")
let nodeInfoContainer = document.getElementById("id-node-information")
let svgExportButton = document.getElementById("id-save-to-svg")
let jsonPrintContainer = document.getElementById("id-json-print-container")
let buttonLoadJsonData = document.getElementById("id-button-load-json")
let fromTopkSelector = document.getElementById("id-topk-range-input")
let formSelectContrast = document.getElementById("id-contrast-selector" );
let formSelectUnivMeasure = document.getElementById("id-univ-measure-selector");
let topkLivePrintContainer = document.getElementById("id-topk-live-print");
let linearInterpolationToPixels = function(value, inputRangeLowerBound, inputRangeUpperBound, outputRangeLowerBound, outputRangeUpperBound){
  // Take value existing in range of [inputRangeLowerBound, inputRangeUpperBound] and casts it into range of
  // [outputRangeLowerBound, outputRangeUpperBound]. This assumes the upper and lower bounds are different in both cases.
  // It is a pure linear interpolation, so no transformation artefacts should be expected.
  // Based on transforming some variable t in range [a,b] into range [c,d] using formula: f(t) = c + ( ( (d - c) / (b - a) ) * (t - a) )
  // This function is used to transform statistical measures in a particular range to statistical measures that map well to pixel sizes
  // Note that output needs to be rounded off to fit into px measures, leading to cutoffs for exponentially or logarithmically behaved variables.
  // The range between a and b is basically divided equally into pixel steps.
  out = Math.round(outputRangeLowerBound + ((outputRangeUpperBound - outputRangeLowerBound) / (inputRangeUpperBound - inputRangeLowerBound) * (value - inputRangeLowerBound)))
  return 
}

function initializeInteractiveNetworkSession(nodes, edges, groups, groupStats) {
  // Constants
  const colorHighlight = "rgb(235,16,162, 0.8)"// "#EB10A2"
  const colorUpRegulated = "#EBB34B"
  const colorDownRegulated = "#6EA5EB"
  const defaultEdgeColor = "#A65A8D"
  const defaultNodeColor = "rgb(166, 90, 141, 0.5)" // "#A65A8D"
  const defaultNodeBorderColor = "rgb(0, 0, 0, 0.7)"

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
      size: 25, font: {size: 14, face: "Helvetica"}, borderWidth: 2, 
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
      let xSum = 0;
      let ySum = 0;
      let count = 0;
      for (node of nodes){
        xSum = xSum + node.x;
        ySum = ySum + node.y;
        count = count + 1;
      }
      let xMean = xSum / count;
      let yMean = ySum / count;
      // Assess current scale of x and y values
      // Set scale to unit scale and multiply with scalingValue
      for (node of nodes){
        node.x = (node.x / xMean) * scalingValue; 
        node.y = (node.y / yMean) * scalingValue;
      }
    }
  }
  let fullEdgeData = new vis.DataSet(edges);
  let networkNodeData = new vis.DataSet();
  let networkEdgeData = new vis.DataSet();
  networkNodeData.add(nodes)
  networkEdgeData.add([])
  let networkData = {nodes: networkNodeData, edges: networkEdgeData}
  

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

  // construct network variable and attach to div
  network = new vis.Network(networkContainer, networkData, networkDrawingOptions);

  // Define network Event Targets, Event Listeners, and Event Handlers:
  network.on("click", function (params) {
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
    
    if (params.nodes.length > 0){
      //params.event = "[original event]";
      let selectedNode = params.nodes[0] // assumes only single selections possible!
      
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
          let topK = fromTopkSelector.value; // <- fix with global setting
          let nElemetsToSelect = Math.min(allEdgesForNode.length, topK);
          topKEdgesForNode = allEdgesForNode.slice(0, nElemetsToSelect)
          return topKEdgesForNode
        }

        return Array.from(filteredEdgeSet)
      }
      let edgeSubset = filterEdges(edges, selectedNode)
      networkEdgeData.update(edgeSubset)
      let nodeGroup = getNodeGroup(network, selectedNode)
      let infoString;
      
      getNodeStatsInfo = function (inputNodeData, selectedNodeId){
        // function expects selectedNode["data"] information
        let outputString = 'Univariate Data for clicked node with id = ' + String(selectedNodeId) + "\n";
        console.log("Checking nodeData", inputNodeData)
        for (const contrastKey in inputNodeData) {
          outputString += `[${contrastKey}:]\n`
          for (const measureKey in inputNodeData[contrastKey]){

            rounded_value = inputNodeData[contrastKey][measureKey]["measure"].toFixed(4)
            outputString += `  ${measureKey}: ${rounded_value}\n`
          }
        }
        return outputString
      }

      getNodeGroupInfo = function (inputGroupData, groupId){
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
      infoGroupLevel = getNodeGroupInfo(groupStats[nodeGroup], nodeGroup)

      resetGroupDrawingOptions(networkDrawingOptions, defaultNodeColor);
      highlightTargetGroup(networkDrawingOptions, nodeGroup, colorHighlight)



      network.storePositions();
      var clickedNode = networkNodeData.get(selectedNode);
      // infoString = JSON.stringify(clickedNode["data"], undefined, 2)
      infoString = getNodeStatsInfo(clickedNode["data"], selectedNode)
      nodeInfoContainer.innerText = infoString + infoGroupLevel;
      network.setOptions(networkDrawingOptions);
      network.redraw();
    } else {
      networkEdgeData.clear();
      resetGroupDrawingOptions(networkDrawingOptions, defaultNodeColor);
      network.setOptions(networkDrawingOptions);
      network.redraw();
    }
  });



  let scalingInput = document.getElementById("id-graph-scaler")
  scalingInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      // code for enter
      resizeLayout(scalingInput.value) // updates the node data
      networkNodeData.clear()
      networkNodeData.update(nodes)
      network.redraw()
      network.fit();
    }
  });

  let runNetworkPhysicsInput = document.getElementById("id-network-physics")
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

  window.onresize = function() {network.fit();}
}

//initializeInteractiveNetworkSession();


// ---------------------------------------------------------------------------------------------------------------------
// Loading new json data and attaching to dom element

//let globalJsonFilePathNodes = "./mockNodes.json"
//let globalJsonFilePathEdges = "./mockEdges.json"

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

    // TODO
    // Define json input schema in a programmatically recognizeable format
    // assert here that the input data conforms with said schema
    // abort data loading if not conform with dashboard expectations. 
    // once implemented, remove the console.assert components below.

    let nodes = inputData["nodes"];
    let edges = inputData["edges"];
    let groupKeys = inputData["groupKeys"];
    let univMeasureKeys = inputData["univMeasureKeys"];
    let groupMeasureKeys = inputData["groupMeasureKeys"];
    let contrastKeys = inputData["contrastKeys"];
    let groupStats = inputData["groupStats"]

    // Processing Edges
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
      let adjustNodeDataToSelection = function (){
        let selectedContrast = domElementContrast.value;
        let selectedMeasure = domElementMeasure.value;
        console.log("Selected Configuration:", selectedContrast, selectedMeasure)
        for (let node of nodes){
          // Replaces the starting node size with the node size of the selected Contrast and Measure. 
          // There is always at least one contrast and one measure to select.
          // For each measure and contrast combination, an appropriate nodeSize value is assumed available
          node["size"] = node["data"][selectedContrast][selectedMeasure]["nodeSize"]
        }
        initializeInteractiveNetworkSession(nodes, edges, groupKeys, groupStats)
      }
      adjustNodeDataToSelection(); // run on init to make sure size properties of nodes align with form element.
      formSelectContrast.addEventListener("change",adjustNodeDataToSelection);
      formSelectUnivMeasure.addEventListener("change",adjustNodeDataToSelection);
      
    }
    initializeContrastMeasureSelectors(formSelectContrast, formSelectUnivMeasure, contrastKeys, univMeasureKeys, nodes, groupKeys)
    initializeInteractiveNetworkSession(nodes, edges, groupKeys, groupStats);
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

C2S.prototype.circle = CanvasRenderingContext2D.prototype.circle;
C2S.prototype.square = CanvasRenderingContext2D.prototype.square;
C2S.prototype.triangle = CanvasRenderingContext2D.prototype.triangle;
C2S.prototype.triangleDown = CanvasRenderingContext2D.prototype.triangleDown;
C2S.prototype.star = CanvasRenderingContext2D.prototype.star;
C2S.prototype.diamond = CanvasRenderingContext2D.prototype.diamond;
C2S.prototype.roundRect = CanvasRenderingContext2D.prototype.roundRect;
C2S.prototype.ellipse_vis = CanvasRenderingContext2D.prototype.ellipse_vis;
C2S.prototype.database = CanvasRenderingContext2D.prototype.database;
C2S.prototype.arrowEndpoint = CanvasRenderingContext2D.prototype.arrowEndpoint;
C2S.prototype.circleEndpoint = CanvasRenderingContext2D.prototype.circleEndpoint;
C2S.prototype.dashedLine = CanvasRenderingContext2D.prototype.dashedLine;

function exportSvg(){
  // function adapted from https://github.com/justinharrell/vis-svg/tree/master
  // var networkContainer = network.body.container; // this is a global variable in my implementation
  var ctx = new C2S({width: networkContainer.clientWidth * 4, height: networkContainer.clientWidth * 4, embedImages: true});
  var canvasProto = network.canvas.__proto__;
  var currentGetContext = canvasProto.getContext;
  canvasProto.getContext = function(){
    return ctx;
  }
  console.log("Checkpoint")
  var svgOptions = {
    nodes: {
      shapeProperties: {interpolation: false},//so images are not scaled svg will get full image
      scaling: { label: { drawThreshold : 0} },
      font:{color:'#000000'}
    },
    edges: {scaling: { label: { drawThreshold : 0} }}
  };
  console.log("Checkpoint")
  //network.setOptions(svgOptions);
  console.log("Checkpoint")
  network.redraw();
  console.log("Checkpoint")
  //network.setOptions(options);
  canvasProto.getContext = currentGetContext;
  ctx.waitForComplete(function()
      {
          var svg = ctx.getSerializedSvg();
          showSvg(svg);
      });
}

function showSvg(svg){
  // function adapted from https://github.com/justinharrell/vis-svg/tree/master
  var svgBlob = new Blob([svg], {type: 'image/svg+xml'});
  openBlob(svgBlob, "network.svg");
}

function openBlob(blob, fileName){
  // function adapted from https://github.com/justinharrell/vis-svg/tree/master
  if(window.navigator && window.navigator.msSaveOrOpenBlob){
    //blobToDataURL(blob, function(dataurl){window.open(dataurl);});
    window.navigator.msSaveOrOpenBlob(blob,fileName);
  }
  else {
    var a = document.getElementById("blobLink");
    if(!a){
      a = document.createElement("a");
      document.body.appendChild(a);
      a.setAttribute("id", "blobLink");
      a.style = "display: none";
    }
    var data = window.URL.createObjectURL(blob);
    a.href = data;
    a.download = fileName;
    a.click();
    // For Firefox it is necessary to delay revoking the ObjectURL
    setTimeout(function(){window.URL.revokeObjectURL(data)}, 100);
  }
}

svgExportButton.addEventListener(type = "click", exportSvg)

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