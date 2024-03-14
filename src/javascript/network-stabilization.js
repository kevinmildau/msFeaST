/** In place coordinates modification of network node coordinates using network physics.
 * 
 * @param {*} keyInput 
 * @param {*} network 
 * @param {*} networkEdgeData 
 * @param {*} fullEdgeData 
 */
let networkStabilizationController = function(keyInput, network, networkEdgeData, fullEdgeData){
  if (keyInput.key === 'Enter') {
    networkEdgeData.update(fullEdgeData);
    let n_iterations = Number(runNetworkPhysicsInput.value);
    network.stabilize(n_iterations);
    networkEdgeData.clear()
    network.storePositions();
    network.redraw()
    network.fit();
  }
  return undefined;
}