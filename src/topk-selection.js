/*
topKSelectionController
*/
topKSelectionController = (function(){
  const rangeInputTopKSelection = document.getElementById("id-topk-range-input");
  const topkLivePrintContainer = document.getElementById("id-topk-live-print");
  let updateTopKTextView = function(){
    topkLivePrintContainer.innerHTML = `Selected top-K value: ${this.value}`;
  };
  rangeInputTopKSelection.addEventListener("change", updateTopKTextView);

  let getTopKValue = function(){
    let value = rangeInputTopKSelection.value;
    return value
  };
  return {updateTopKTextView : updateTopKTextView, getTopKValue : getTopKValue};
})();



