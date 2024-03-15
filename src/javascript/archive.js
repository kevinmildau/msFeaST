/**
 * Take value existing in range of [inputRangeLowerBound, inputRangeUpperBound] and casts it into range of
 * [outputRangeLowerBound, outputRangeUpperBound]. This assumes the upper and lower bounds are different in both cases.
 *
**/
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