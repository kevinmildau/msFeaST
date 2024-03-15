from dataclasses import dataclass
from typing import List, Union

@dataclass
class GridEntryTsne:
  """ 
  Container Class for t-SNE embedding optimization results. Contains a single entry. A list of these containers can be 
  converted to pandas for easy display.

  Parameters:
    perplexity : int with perplexity value used in t-SNE optimization.
    x_coordinates : List[int] x coordinates produced by t-SNE
    y_coordinates:  List[int] y coordinates produced by t-SNE
    pearson_score : float representing the pearson correlation between pairwise distances in embedding and 
      high dimensional space.
    spearman_score : float representing the spearman correlation between pairwise distances in embedding and 
      high dimensional space.
    random_seed_used : int or float with the random seed used in k-medoid clustering.
  """
  perplexity : int
  x_coordinates : List[int]
  y_coordinates:  List[int]
  pearson_score : float
  spearman_score : float
  random_seed_used : Union[int, float]
  def __str__(self) -> str:
    custom_print = (
      f"Perplexity = {self.perplexity}," 
      f"Pearson Score = {self.pearson_score}, "
      f"Spearman Score = {self.spearman_score}, \n"
      f"x coordinates = {', '.join(self.x_coordinates[0:4])}...",
      f"y coordinates = {', '.join(self.y_coordinates[0:4])}...")
    return custom_print

@dataclass
class GridEntryUmap:
  """ 
  NOT IMPLEMENTED
  Results container for umap grid computation. A list of these containers can be converted to pandas for easy display.
  """
  ...

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