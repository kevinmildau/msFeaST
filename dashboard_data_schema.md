JSON input data structure for dashboard:

The dashboard json structure consists of a nested infromative key chain. The first layer contains the following keys:

- nodes
- groupStats
- edges
- univMeasureKeys
- groupMeasureKeys
- contrastKeys

The last thee keys contain the respective unique keys for univariate measures, group measures, and contrasts as arrays 
strings. Edges contains edges connecting nodes via node identifiers. Nodes and groupStats contain node and group 
informaiton. Group stats immediately nests into contrasts keys, each populated by measure keys, each with a 
corresponding measure value. Nodes contains top level node informaiton such as coordinates, and a data key which
nests into contrast keys, containing measure keys, containing each measure value and measure NodeSize keys, each with
corresponding values. For an abstracted complete example see below:

```json
{
	// groupKeys provides a quick reference for the number of groups and their keys
	groupKeys: ["groupKey1", "groupKey2", ...],
	// groupMemberships collects the feature ids belonging to each groupKey as an array of strings
	groupMemberships: {"groupKey1" : ["uniqueNodeIdentifier1", "uniqueNodeIdentifier2"], ...},
	// univMeasureKeys provides a quick reference for the number of univ. measures and their keys
	univMeasureKeys: ["univMeasureKey1", "univMeasureKey2", ...],
	// groupMeasureKeys provides a quick reference for the number of group based. measures and their keys
	groupMeasureKeys: ["groupMeasureKey1", "groupMeasureKey2", ...],
	// contrastKeys provides a quick reference for the number of contrasts and their keys
	contrastKeys: ["contrastKey1", "contrastKey2", ...],
	// groupStats provides statistical summary information for each groupKey in groupKeys. The number of entries
	// equals the number of groupMeasureKeys.
	groupStats: {
		"groupKey1": {
			"contrastKey1": {
				"groupMeasureKey1": value,
				"groupMeasureKey2": value
			},
			"contrastKey2": {
				"groupMeasureKey1": value,
				"groupMeasureKey2": value
			},
		},
		"groupKey1": {
			"contrastKey1": {
				"groupMeasureKey1": value,
				"groupMeasureKey2": value
			},
			"contrastKey2": {
				"groupMeasureKey1": value,
				"groupMeasureKey2": value
			},	
		}, ...
	},
	nodes: [
		{
			"id": "uniqueNodeIdentifier", // unique for each node in the network
			"size": numericSize, // A node size in px. Based on a univ. statistical measure scaled to pixels between 10 and 50.
			"group": "groupKey1", // matching the keys in groupNames exactly
			"x": numericCoordinateValue, // x location on canvas for the node in pixels
			"y": numericCoordinateValue, // y location on canvas for the node in pixels
			"data" : { // data is an generic term for the json structured statistical info for each node.
				"spectrum_ms_information": {
					precursor_mz : value, // precursor mz value from the .mgf file
					retention_time : value, // ms1 retention time information, in the unit used within the .mgf file
				},
				// all node specific information goes here, ordered by contrast
				"contrastKey1": {
					"univMeasureKey1": {"measure": value, "nodeSize": value}, // the measure value can be arbitrary, the nodeSize value should be in range 10 to 50 based on the measure value
					"univMeasureKey2": {"measure": value, "nodeSize": value}, // the measure value can be arbitrary, the nodeSize value should be in range 10 to 50 based on the measure value
					...
				},
				"contrastKey2": {
					"univMeasureKey1": {"measure": value, "nodeSize": value}, // the measure value can be arbitrary, the nodeSize value should be in range 10 to 50 based on the measure value
					"univMeasureKey2": {"measure": value, "nodeSize": value}, // the measure value can be arbitrary, the nodeSize value should be in range 10 to 50 based on the measure value
					...
				},
				...
			}
		},
	],
	edges: [
		"id": "edgeKey", // a unique key for each edge across all edges
		// from and to are generic endpoints since we don't make use directed edges. 
		// note that, if a node is pointed to by many other nodes, its number of edges may exceed the set 
		// number. This is a consequence of top-K edges being selected for each node separately.
		"from": "uniqueNodeIdentifier1", // a unique node identifier from which the edge originates 
		"to": "uniqueNodeIdentifier2", // a unique node identifier to which the edge points
		"width": value, // the similarity score projected to the range between 1px and 30px
		"data": {
			"score": value // the actual value of the score to be displayed as an edge label (up to some rounding)
			... // Possible additional data keys for edges currently not read but could be added in future updates.
		} 
	]
}
```
