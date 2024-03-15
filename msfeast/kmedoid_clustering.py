from dataclasses import dataclass
from kmedoids import KMedoids
from sklearn.metrics import silhouette_score
from typing import List, Union

@dataclass
class GridEntryKmedoid:
    """ 
    Container Class for K medoid clustering results. Contains a single entry. A list of these containers can be
    converted to pandas for easy display.

    Parameters:
        k: the number of clusters set.
        cluster_assignments: List with cluster assignment for each observation.
        silhouette_score: float with clustering silhouette score.
        random_seed_used : int or float with the random seed used in k-medoid clustering.
    """
    k : int
    cluster_assignments : List[int]
    silhouette_score : float
    random_seed_used : Union[int, float]
    def __str__(self) -> str:
        """ Custom Print Method for kmedoid grid entry producing an easy readable string output. """
        custom_print = (
            f"k = {self.k}, silhoutte_score = {self.silhouette_score}, \n"
            f"cluster_assignment = {', '.join(self.cluster_assignments[0:7])}...")
        return custom_print


def run_kmedoid_grid(
    distance_matrix : np.ndarray, 
    k_values : List[int], 
    random_states : Union[List, None] = None
    ) -> List[GridEntryKmedoid]:
  """ Runs k-medoid clustering for every value in k_values. 
  
  Parameters:
      distance_matrix: An np.ndarray containing pairwise distances.
      k_values: A list of k values to try in k-medoid clustering.
      random_states: None or a list of integers specifying the random state to use for each k-medoid run.
  Returns: 
      A list of GridEntryKmedoid objects containing grid results.
  """
  if random_states is None:
      random_states = [ 0 for _ in k_values ]
  output_list = []
  check_k_values(k_values, max_k = distance_matrix.shape[0])
  for idx, k in enumerate(k_values):
      cluster = KMedoids(
          n_clusters=k, 
          metric='precomputed', 
          random_state=random_states[idx], 
          method = "fasterpam"
      )  
      cluster_assignments = cluster.fit_predict(distance_matrix)
      cluster_assignments_strings = [
          "km_" + str(elem) 
          for elem in cluster_assignments
      ]
      score = silhouette_score(
          X = distance_matrix, 
          labels = cluster_assignments_strings, 
          metric= "precomputed"
      )
      output_list.append(
          GridEntryKmedoid(
              k, 
              cluster_assignments_strings, 
              score, 
              random_states[idx]
          )
      )
  return output_list

def check_k_values(k_values : List[int], max_k : int) -> None:
    """ Function checks whether k values match expected configuration. Aborts if not. """
    assert k_values is not [], (
        "Error: k_values list is empty! This may be a result of post-processing: there must be a "
        "k value below the number of features/spectra for optimization to work."
    )
    assert isinstance(k_values, list), (
        "Error: k_values must be a list. If only running one value, specify input as [value]."
    )
    for k_value in k_values: 
        assert isinstance(k_value, int) and k_value < max_k, (
            "Error: k_value must be numeric (int) and smaller than number of features/spectra." 
        )
    return None

def _print_kmedoid_grid(grid : List[GridEntryKmedoid]) -> None:
  """ Prints all values in kmedoid grid in readable format via pandas conversion """
  kmedoid_results = pd.DataFrame.from_dict(data = grid).loc[
      :, ["k", "silhouette_score", "random_seed_used"]
  ]
  kmedoid_results.insert(loc = 0, column = "iloc", value = [iloc for iloc in range(0, len(grid))])
  print("Kmedoid grid results. Use to inform kmedoid classification selection ilocs.")
  print(kmedoid_results)
  return None

def _plot_kmedoid_grid(
  kmedoid_list : List[GridEntryKmedoid]
  ) -> None:
  """ Plots Silhouette Score vs k for each entry in list of GridEntryKmedoid objects. """
  scores = [x.silhouette_score for x in kmedoid_list]
  ks = [f"k = {x.k} / iloc = {iloc}" for iloc, x in enumerate(kmedoid_list)]
  fig = plotly.express.scatter(x = ks, y = scores)
  fig.update_layout(
    xaxis_title="K (Number of Clusters) / iloc", 
    yaxis_title="Silhouette Score"
  )
  fig.show()
  return None