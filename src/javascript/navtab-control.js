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

