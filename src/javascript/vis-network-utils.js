/** Function filters edge list down to 
 * 
 * @param {*} edgeList 
 * @param {*} nodeId 
 * @returns 
 */
const filterEdges = function(edgeList, nodeId){
  let filteredEdgeSet = new Set([]);
  if (edgeList.length > 0){
    for (let i = 0; i < edgeList.length; i++) {
      if (edgeList[i].from == nodeId || edgeList[i].to == nodeId ){
        filteredEdgeSet.add(edgeList[i])
      };
    };
    let allEdgesForNode = Array.from(filteredEdgeSet);
    allEdgesForNode.sort((a,b) => a.data.score - b.data.score).reverse();
    let topK = topKSelectionModule.getTopKValue();
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
const getNodeDataInfo = function (inputNodeData, selectedNodeId, contrastKeys){
  // function expects selectedNode["data"] information
  let outputString = 'Data for clicked node with id: ' + String(selectedNodeId) + "\n";
  //console.log("Checking nodeData", inputNodeData)
  if ("spectrum_ms_information" in inputNodeData){
    outputString += 'Spectrum information:\n'
    outputString += `  precursor_mz: ${inputNodeData["spectrum_ms_information"]["precursor_mz"]}\n`
    outputString += `  retention_time: ${inputNodeData["spectrum_ms_information"]["retention_time"]}\n`
  }
  outputString += 'Univariate data:\n'
  for (const contrastKey of contrastKeys) {
    outputString += `[${contrastKey}:]\n`
    for (const measureKey in inputNodeData[contrastKey]){
      if (["univariatePValue", "log2FoldChange"].includes(measureKey)){
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


/**
 * 
 * @param {*} groupMemberships group_id keyed object with feature_ids for each member
 * @param {*} groupId 
 * @returns string with feature_ids styles to add no more than 5 ids per line of text.
 */
let getGroupmemberships = function (groupMemberships, groupId){
  let outputString = 'Features belonging to group with id =  ' + String(groupId) + ":\n";
  let counter = 1;
  for (const feature_id of groupMemberships[groupId]) {
    if (counter % 10 == 0){
      outputString += "\n";
    }
    outputString += String(feature_id) + ", ";
    counter += 1;
  }
  outputString += "\n";
  return outputString
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