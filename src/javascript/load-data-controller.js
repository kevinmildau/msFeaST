
/** Sets contrast key options in formSelectContrast
 * 
 * @param {*} optionsArrayContrastKeys 
 */
let setContrastKeys = function(optionsArrayContrastKeys){
  formSelectContrast.innerHTML = "";
  optionsArrayContrastKeys.forEach(function(optionKey) {
    let option = document.createElement("option");
    option.value = optionKey;
    option.text = optionKey;
    formSelectContrast.appendChild(option);
  });
  return undefined
}

/** Sets contrast key options in formSelectUnivMeasure
 * 
 * @param {*} optionsArrayContrastKeys 
 */
let setMeasureKeys = function(optionsArrayMeasureKeys){
  formSelectUnivMeasure.innerHTML = "";
  optionsArrayMeasureKeys.forEach(function(optionKey) {
    let option = document.createElement("option");
    option.value = optionKey;
    option.text = optionKey;
    formSelectUnivMeasure.appendChild(option);
  });
  return undefined
}

/** Function checks whether input json data contains expected keys.
 * 
 * @param {*} jsonData 
 * @returns {boolean} Indicating whether assertions were passed (true) or not (false)
 */
let assertInputSchemaValid = function (jsonData){
  // basic function to check whether the expected entries are available, to be replaced with object schema and type validation
  // jsonData is expected to be type object after a successful json parse. JSON parsing errors are handled separately.
  let schemaValid = true; 
  let expectedKeys = ["groupKeys", "groupMemberships", "univMeasureKeys", "groupMeasureKeys", "contrastKeys", "groupStats", "nodes", "edges"]
  for (key of expectedKeys){
    if (typeof(jsonData[key]) === "undefined"){
      schemaValid = false; 
      // The console log below is part of the desired console output
      console.log(`Key entry "${key}" not found in input data.`);
    }
  }
  if (schemaValid === false){
    alert(
      "Provided file does not contain expected input structues. See console log for missing entry." + 
      "Aborting file loading and interactive session initialization."
    );
  }
  return schemaValid
}

let parseInputAndInitalizeDashboard = function (fileInput) {
  let textLines = fileInput.target.result;
  let jsonData;
  let nodes, edges, groupKeys, groupStats, univMeasureKeys, contrastKeys;
  // Attempt Parsing of file input to json
  
  try {
    jsonData = JSON.parse(textLines);
  } catch (error) {
    alert(
      `Input data does not appear to conform with JSON file standard.` + 
      `Aborting data loading. Following error was generated: ${error}`
    );
    return undefined
  }
  // 
  if(!assertInputSchemaValid(jsonData)){return undefined;}
  nodes = jsonData["nodes"];
  edges = jsonData["edges"];
  groupKeys = jsonData["groupKeys"];
  groupMemberships = jsonData["groupMemberships"];
  groupStats = jsonData["groupStats"];
  univMeasureKeys = jsonData["univMeasureKeys"];
  contrastKeys = jsonData["contrastKeys"];
  //let groupMeasureKeys = jsonData["groupMeasureKeys"];
  
  setContrastKeys(contrastKeys);
  setMeasureKeys(univMeasureKeys);
  initializeInteractiveVisualComponents(nodes, edges, groupKeys, groupStats, contrastKeys, groupMemberships);
};

let assertFileReaderSupport = function (){
  let fileReaderSupported = true;
  if (typeof window.FileReader !== 'function') {
    alert("The file API isn't supported on this browser yet.");
    fileReaderSupported = false;
    return fileReaderSupported;
  } else {
    return fileReaderSupported;
  };
};

let assertFilePropertySupport = function (input){
  let filePropertySupported = true;
  if (input.files === "undefined") {
    alert("This browser does not seem to support the `files` property of file inputs.");
    filePropertySupported = false;
    return filePropertySupported;
  } else {
    return filePropertySupported;
  };
};

/** Reads and parses json data and calls dashboard initialization modules.
 * 
 */
let dataLoadingController = function() {
  let file, fileReader;
  let domElementFileInput = document.getElementById('id-file-input');
  if (!assertFileReaderSupport()){return;};
  if (!assertFilePropertySupport(domElementFileInput)){return;};
  if (domElementFileInput.files[0] === "undefined") {
    alert("Please select a file before clicking 'Load'"); 
    return;
  } else {
    file = domElementFileInput.files[0];
    fileReader = new FileReader();
    fileReader.addEventListener("load", parseInputAndInitalizeDashboard);
    fileReader.readAsText(file);
  };
};

