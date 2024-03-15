let heatmapPanelController = function(groupStats, domElementContrast, networkDrawingOptions, network){
  // Constructs heatmap with global test set specific results
  // Dev Note: Assumes n_measures at set level is equal to 1!
  let colorscale = 'YlGnBu';
  let margin = 10;
  /** Function constructs heatmap trace and layout from groupStats data
   *  
   * @param {*} groupStats 
   * @returns 
   */
  let constructHeatmapData = function(groupStats, colorscale, margin){
    let xTicks = Object.keys(groupStats);
    let yTicks = Array.from(domElementContrast.options).map(option => option.value); // Change global access to local data!
    // get array of p-values for all constrat
    let globalTestPValuesArray = [];
    for (let current_contrast of yTicks) {
      tmp_array = [];
      for (let current_group of xTicks) {
        tmp_array.push(groupStats[current_group][current_contrast]["globalTestPValue"]);
      }
      globalTestPValuesArray.push(tmp_array);
    }
    let heatmapTrace = [{
      z: globalTestPValuesArray,
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
    var heatmapLayout = {
      margin: {t:margin, l:margin},
      xaxis: {automargin : true, tickmode: "linear", dtick:1, tickfont : {size : 8}},
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
  heatmapObject = constructHeatmapData(groupStats, colorscale, margin);
  colorBarObject = constructColorBarData(colorscale, margin);
  updateViews(heatmapObject, colorBarObject);
  var hoverTimer;  // Define a timer variable
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
        resetGroupDrawingOptions(networkDrawingOptions, stylingVariables.defaultNodeColor) // autoreset at every hover
        highlightTargetGroup(networkDrawingOptions, xValue, stylingVariables.colorHighlight)
        network.setOptions(networkDrawingOptions);
        network.redraw();
      }, 
      delay_hover
    );
  });
}