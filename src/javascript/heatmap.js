let heatmapPanelController = function(groupStats, domElementContrast, networkDrawingOptions, network, highlightGroupId = undefined){
  // Constructs heatmap with global test set specific results
  // Dev Note: Assumes n_measures at set level is equal to 1!
  let colorscale = 'Greys'; // Greys, Rainbow, Portland (diverging), Jet, Hot has clear low end separation
  let margin = 10;

  /** Function creates xTicks with highlight style if provided a highlight group
   *  
   * @param {*} groupIdentifiers array of identifiers 
   * @param {*} highlightGroup the identifier to be highlighted in the axis text
   * @returns 
   */
  let generateXaxisTicks = function(groupIdentifiers, highlightGroup){
    let xTicks = [];
    for (tick of groupIdentifiers){
      if (tick === highlightGroup){
        xTicks.push(`<span style='font-weight:bold;font-size:100%;color:${stylingVariables["colorHighlight"]}'>${String(tick)}</span>`)
      } else {
        xTicks.push(tick)
      }
    }
    return xTicks
  }
  /** Function constructs heatmap trace and layout from groupStats data
   *  
   * @param {*} groupStats 
   * @returns 
   */
  let constructHeatmapData = function(groupStats, colorscale, margin, highlightGroupId){
    let xVals = Object.keys(groupStats);
    let xTicks = generateXaxisTicks(xVals, highlightGroupId);
    let yTicks = Array.from(domElementContrast.options).map(option => option.value); // Change global access to local data!
    // get array of p-values for all constrat
    let globalTestPValuesArray = [];
    for (let current_contrast of yTicks) {
      tmp_array = [];
      for (let current_group of xVals) {
        tmp_array.push(groupStats[current_group][current_contrast]["globalTestPValue"]);
      }
      globalTestPValuesArray.push(tmp_array);
    }
    let heatmapTrace = [{
      z: globalTestPValuesArray,
      x: xVals,
      y: yTicks,
      zmin: 0,  // Minimum color value
      zmax: 1,   // Maximum color value
      colorscale: colorscale, // [[0, 'white'], [1, 'blue']], // for custom color scale
      showscale : false,
      //reversescale : true,
      type: 'heatmap',
      hovertemplate: 'x: %{x}<br>y: %{y}<br>z: %{z:.2e}<extra></extra>'
    }];
    var heatmapLayout = {
      margin: {t:margin, l:margin},
      xaxis: {automargin : true, tickmode: "array", dtick:1, tickfont : {size : 8} ,tickvals: xVals, ticktext : xTicks}, // 
      yaxis: {automargin : true, side: "right"},
      title: '',
      autosize: true,
    };
    return {trace : heatmapTrace, layout: heatmapLayout};
  };

  /** Functions returns colorbar trace and layout for heatmap. Currently largely a static object constructor.
   */
  const constructColorBarData = function(colorscale, margin){
    let scale_values = Array.from({length: 101}, (_, i) => i * 0.01);
    let colorBarTrace = [{
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
    let colorBarLayout = {
      title : "",
      margin: {l: margin,r: margin,b: margin, t: margin}, 
      xaxis: {automargin : true, domain: [0, 1], tickfont : {size : 8}, anchor: 'y'},
      yaxis: {anchor: 'x', automargin : true, showticklabels: true, side: "right", tickmode: "linear", dtick:1},
      autosize: true, // Automatically adjust size to container
    };
    return {trace : colorBarTrace, layout: colorBarLayout};
  };
  let updateViews = function(heatmapObject, colorBarObject){
    Plotly.newPlot(colorBarContainer, colorBarObject.trace, colorBarObject.layout, {responsive : true});
    Plotly.newPlot(heatmapContainer, heatmapObject.trace, heatmapObject.layout, {responsive : true});
  };
  heatmapObject = constructHeatmapData(groupStats, colorscale, margin, highlightGroupId);
  colorBarObject = constructColorBarData(colorscale, margin);
  updateViews(heatmapObject, colorBarObject);
  //var hoverTimer;  // Define a timer variable
  //var delay_hover = 0; // Delay in milliseconds
  heatmapContainer.on('plotly_click', data => updateUsingClickData(data, networkDrawingOptions, network));
}

/** Takes plotly click data and updates the network visualization highlight group accordingly.
 * 
 * @param {data} plotly click data 
 */
let updateUsingClickData = function(data, networkDrawingOptions, network){
  var xValue = data.points[0].x;
  resetGroupDrawingOptions(networkDrawingOptions, stylingVariables.defaultNodeColor) // autoreset at every hover
  highlightTargetGroup(networkDrawingOptions, xValue, stylingVariables.colorHighlight)
  network.setOptions(networkDrawingOptions);
  network.redraw();
}