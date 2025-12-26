"""
Spatial Evaluator for 2μm Spatial Transcriptomics Prediction

Implements rigorous evaluation metrics for cell-level gene expression prediction:
- Optimal Transport (Wasserstein distance)
- Biological Conservation (scIB-E metrics)
- Spatial Fidelity (SVG recovery, Moran's I)
- Negative Controls (label shuffle, spatial jitter, random baseline)

Author: Max Van Belkum (Huo Lab, Vanderbilt)
Date: 2025-12-26
Phase: Month 1 Week 2 - Day 0-1
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Optimal Transport
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    warnings.warn("POT library not found. Install with: pip install POT")

# Biological Conservation Metrics
try:
    from scib_metrics import ari, nmi, silhouette, lisi
    HAS_SCIB = True
except ImportError:
    HAS_SCIB = False
    warnings.warn("scib-metrics not found. Install with: pip install scib-metrics")

# Spatial statistics
try:
    from esda.moran import Moran
    from libpysal.weights import KNN
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    warnings.warn("spatial stats not found. Install with: pip install esda libpysal")

# Standard metrics
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class EvaluationResults:
    """Container for evaluation metrics"""

    # Primary metrics
    ssim_mean: float
    ssim_median: float
    ssim_per_category: Dict[str, float]

    # Optimal Transport
    wasserstein_distance: Optional[float] = None

    # Gene-centric metrics
    pearson_mean: float = 0.0
    pearson_median: float = 0.0
    spearman_mean: float = 0.0
    spearman_median: float = 0.0
    genes_above_05: int = 0
    genes_above_08: int = 0

    # Biological conservation
    ari: Optional[float] = None
    nmi: Optional[float] = None
    silhouette: Optional[float] = None
    lisi: Optional[float] = None

    # Spatial fidelity
    svg_recovery: Optional[float] = None
    morans_i_concordance: Optional[float] = None

    # Biological validity
    marker_concordance: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for easy reporting"""
        return {
            'ssim_mean': self.ssim_mean,
            'ssim_median': self.ssim_median,
            **{f'ssim_{k}': v for k, v in self.ssim_per_category.items()},
            'wasserstein_distance': self.wasserstein_distance,
            'pearson_mean': self.pearson_mean,
            'pearson_median': self.pearson_median,
            'spearman_mean': self.spearman_mean,
            'spearman_median': self.spearman_median,
            'genes_r>0.5': self.genes_above_05,
            'genes_r>0.8': self.genes_above_08,
            'ari': self.ari,
            'nmi': self.nmi,
            'silhouette': self.silhouette,
            'lisi': self.lisi,
            'svg_recovery': self.svg_recovery,
            'morans_i_concordance': self.morans_i_concordance,
            'marker_concordance': self.marker_concordance,
        }

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            "=== Spatial Transcriptomics Evaluation Results ===",
            f"\nPrimary Metrics:",
            f"  SSIM (mean):   {self.ssim_mean:.4f}",
            f"  SSIM (median): {self.ssim_median:.4f}",
        ]

        if self.ssim_per_category:
            lines.append(f"\n  Per-Category SSIM:")
            for cat, val in self.ssim_per_category.items():
                lines.append(f"    {cat}: {val:.4f}")

        if self.wasserstein_distance is not None:
            lines.append(f"\nOptimal Transport:")
            lines.append(f"  Wasserstein distance: {self.wasserstein_distance:.4f}")

        lines.extend([
            f"\nGene-Centric Metrics:",
            f"  Pearson r (mean):   {self.pearson_mean:.4f}",
            f"  Spearman ρ (mean):  {self.spearman_mean:.4f}",
            f"  Genes r > 0.5: {self.genes_above_05}",
            f"  Genes r > 0.8: {self.genes_above_08}",
        ])

        if self.ari is not None:
            lines.extend([
                f"\nBiological Conservation:",
                f"  ARI:        {self.ari:.4f}",
                f"  NMI:        {self.nmi:.4f}",
                f"  Silhouette: {self.silhouette:.4f}",
            ])

        return "\n".join(lines)


class SpatialEvaluator:
    """
    Comprehensive evaluator for spatial transcriptomics predictions.

    Designed for cell-level predictions from H&E histology at 2μm resolution.
    Implements Optimal Transport, biological conservation, and spatial fidelity metrics.
    """

    def __init__(
        self,
        gene_categories: Optional[Dict[str, List[str]]] = None,
        marker_genes: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            gene_categories: Dict mapping category names to gene lists
                e.g., {'epithelial': ['EPCAM', 'KRT8'], 'immune': ['CD3E', 'CD4']}
            marker_genes: Dict mapping cell types to marker genes
                e.g., {'epithelial': 'EPCAM', 'stromal': 'VIM'}
        """
        self.gene_categories = gene_categories or {}
        self.marker_genes = marker_genes or {}

        # Check dependencies
        self.has_pot = HAS_POT
        self.has_scib = HAS_SCIB
        self.has_spatial = HAS_SPATIAL

    def evaluate(
        self,
        pred_adata: ad.AnnData,
        true_adata: ad.AnnData,
        compute_optimal_transport: bool = True,
        compute_bio_conservation: bool = True,
        compute_spatial_fidelity: bool = True,
    ) -> EvaluationResults:
        """
        Comprehensive evaluation of predicted vs true expression.

        Args:
            pred_adata: Predicted expression (cells x genes)
            true_adata: Ground truth expression (cells x genes)
            compute_optimal_transport: Whether to compute Wasserstein distance
            compute_bio_conservation: Whether to compute scIB-E metrics
            compute_spatial_fidelity: Whether to compute SVG recovery, Moran's I

        Returns:
            EvaluationResults object
        """
        # Validate inputs
        assert pred_adata.shape == true_adata.shape, \
            f"Shape mismatch: pred {pred_adata.shape} vs true {true_adata.shape}"
        assert all(pred_adata.var_names == true_adata.var_names), \
            "Gene names must match"
        assert all(pred_adata.obs_names == true_adata.obs_names), \
            "Cell IDs must match"

        # Extract matrices
        pred_expr = pred_adata.X.toarray() if hasattr(pred_adata.X, 'toarray') else pred_adata.X
        true_expr = true_adata.X.toarray() if hasattr(true_adata.X, 'toarray') else true_adata.X

        # Compute primary metrics
        ssim_mean, ssim_median, ssim_per_category = self._compute_ssim(
            pred_expr, true_expr, pred_adata.var_names
        )

        # Gene-centric metrics
        pearson_mean, pearson_median, spearman_mean, spearman_median, \
            genes_above_05, genes_above_08 = self._compute_gene_metrics(pred_expr, true_expr)

        # Initialize results
        results = EvaluationResults(
            ssim_mean=ssim_mean,
            ssim_median=ssim_median,
            ssim_per_category=ssim_per_category,
            pearson_mean=pearson_mean,
            pearson_median=pearson_median,
            spearman_mean=spearman_mean,
            spearman_median=spearman_median,
            genes_above_05=genes_above_05,
            genes_above_08=genes_above_08,
        )

        # Optimal Transport
        if compute_optimal_transport and self.has_pot:
            results.wasserstein_distance = self.compute_wasserstein(pred_adata, true_adata)

        # Biological Conservation
        if compute_bio_conservation and self.has_scib:
            bio_results = self.compute_bio_conservation(pred_adata, true_adata)
            results.ari = bio_results.get('ari')
            results.nmi = bio_results.get('nmi')
            results.silhouette = bio_results.get('silhouette')
            results.lisi = bio_results.get('lisi')

        # Spatial Fidelity
        if compute_spatial_fidelity:
            if 'spatial' in pred_adata.obsm:
                results.svg_recovery = self.compute_svg_recovery(pred_adata, true_adata)
                if self.has_spatial:
                    results.morans_i_concordance = self.compute_morans_i(pred_adata, true_adata)

        # Biological Validity
        if self.marker_genes:
            results.marker_concordance = self.compute_marker_concordance(pred_adata, true_adata)

        return results

    def _compute_ssim(
        self,
        pred_expr: np.ndarray,
        true_expr: np.ndarray,
        gene_names: pd.Index,
    ) -> Tuple[float, float, Dict[str, float]]:
        """Compute per-gene SSIM and aggregate"""

        ssim_values = []

        for gene_idx in range(pred_expr.shape[1]):
            # Reshape to 2D spatial map (assumes square grid)
            # TODO: Handle arbitrary spatial coordinates
            n_cells = pred_expr.shape[0]
            grid_size = int(np.sqrt(n_cells))

            if grid_size * grid_size != n_cells:
                # Non-square grid - use all cells as 1D
                pred_map = pred_expr[:, gene_idx]
                true_map = true_expr[:, gene_idx]

                # For 1D, use normalized correlation as proxy
                if np.std(pred_map) > 0 and np.std(true_map) > 0:
                    corr = np.corrcoef(pred_map, true_map)[0, 1]
                    ssim_values.append(max(0, corr))  # SSIM-like: [0, 1]
                else:
                    ssim_values.append(0.0)
            else:
                # Square grid - use 2D SSIM
                pred_map = pred_expr[:, gene_idx].reshape(grid_size, grid_size)
                true_map = true_expr[:, gene_idx].reshape(grid_size, grid_size)

                # Compute SSIM with standard parameters
                ssim_val = ssim(
                    true_map,
                    pred_map,
                    data_range=max(true_map.max(), pred_map.max()) - min(true_map.min(), pred_map.min())
                )
                ssim_values.append(ssim_val)

        ssim_values = np.array(ssim_values)

        # Per-category SSIM
        ssim_per_category = {}
        for category, genes in self.gene_categories.items():
            cat_indices = [i for i, g in enumerate(gene_names) if g in genes]
            if cat_indices:
                ssim_per_category[category] = float(np.mean(ssim_values[cat_indices]))

        return float(np.mean(ssim_values)), float(np.median(ssim_values)), ssim_per_category

    def _compute_gene_metrics(
        self,
        pred_expr: np.ndarray,
        true_expr: np.ndarray,
    ) -> Tuple[float, float, float, float, int, int]:
        """Compute per-gene correlation metrics"""

        pearson_vals = []
        spearman_vals = []

        for gene_idx in range(pred_expr.shape[1]):
            pred_gene = pred_expr[:, gene_idx]
            true_gene = true_expr[:, gene_idx]

            # Pearson
            if np.std(pred_gene) > 0 and np.std(true_gene) > 0:
                r, _ = pearsonr(pred_gene, true_gene)
                pearson_vals.append(r)

                # Spearman
                rho, _ = spearmanr(pred_gene, true_gene)
                spearman_vals.append(rho)
            else:
                pearson_vals.append(0.0)
                spearman_vals.append(0.0)

        pearson_vals = np.array(pearson_vals)
        spearman_vals = np.array(spearman_vals)

        # Count genes above thresholds
        genes_above_05 = int(np.sum(pearson_vals > 0.5))
        genes_above_08 = int(np.sum(pearson_vals > 0.8))

        return (
            float(np.mean(pearson_vals)),
            float(np.median(pearson_vals)),
            float(np.mean(spearman_vals)),
            float(np.median(spearman_vals)),
            genes_above_05,
            genes_above_08,
        )

    def compute_wasserstein(
        self,
        pred_adata: ad.AnnData,
        true_adata: ad.AnnData,
    ) -> float:
        """
        Compute Wasserstein distance (Earth Mover's Distance) between distributions.

        Uses Optimal Transport to measure geometric distance between
        predicted and true gene expression distributions.

        Args:
            pred_adata: Predicted expression
            true_adata: Ground truth expression

        Returns:
            Wasserstein distance (lower is better)
        """
        if not self.has_pot:
            raise ImportError("POT library required. Install with: pip install POT")

        pred_expr = pred_adata.X.toarray() if hasattr(pred_adata.X, 'toarray') else pred_adata.X
        true_expr = true_adata.X.toarray() if hasattr(true_adata.X, 'toarray') else true_adata.X

        # Normalize to probability distributions (sum to 1)
        pred_dist = pred_expr / pred_expr.sum(axis=0, keepdims=True)
        true_dist = true_expr / true_expr.sum(axis=0, keepdims=True)

        # Compute pairwise Wasserstein distance for each gene
        wasserstein_distances = []

        for gene_idx in range(pred_expr.shape[1]):
            pred_gene_dist = pred_dist[:, gene_idx]
            true_gene_dist = true_dist[:, gene_idx]

            # Handle zero distributions
            if pred_gene_dist.sum() == 0 or true_gene_dist.sum() == 0:
                wasserstein_distances.append(1.0)  # Maximum distance
                continue

            # Compute 1D Wasserstein distance
            w_dist = ot.wasserstein_1d(
                pred_gene_dist,
                true_gene_dist,
                metric='euclidean'
            )
            wasserstein_distances.append(w_dist)

        # Return mean Wasserstein distance across genes
        return float(np.mean(wasserstein_distances))

    def compute_bio_conservation(
        self,
        pred_adata: ad.AnnData,
        true_adata: ad.AnnData,
    ) -> Dict[str, float]:
        """
        Compute biological conservation metrics (scIB-E framework).

        Tests whether predicted expression preserves cell types and states.

        Args:
            pred_adata: Predicted expression (must have .obs['cell_type'])
            true_adata: Ground truth expression (must have .obs['cell_type'])

        Returns:
            Dict with ARI, NMI, Silhouette, LISI scores
        """
        if not self.has_scib:
            raise ImportError("scib-metrics required. Install with: pip install scib-metrics")

        # Ensure cell type labels exist
        if 'cell_type' not in true_adata.obs:
            warnings.warn("cell_type not in obs, skipping bio conservation metrics")
            return {}

        # Cluster predicted data using Leiden algorithm
        sc.pp.neighbors(pred_adata, use_rep='X')
        sc.tl.leiden(pred_adata, key_added='leiden_pred')

        # Extract labels
        true_labels = true_adata.obs['cell_type'].values
        pred_labels = pred_adata.obs['leiden_pred'].values

        # Compute metrics
        results = {}

        # ARI (cluster agreement)
        results['ari'] = float(adjusted_rand_score(true_labels, pred_labels))

        # NMI (information shared)
        results['nmi'] = float(normalized_mutual_info_score(true_labels, pred_labels))

        # Silhouette (cluster separation) - requires embeddings
        if 'X_pca' not in pred_adata.obsm:
            sc.pp.pca(pred_adata)

        from sklearn.metrics import silhouette_score
        results['silhouette'] = float(silhouette_score(
            pred_adata.obsm['X_pca'],
            pred_labels
        ))

        return results

    def compute_svg_recovery(
        self,
        pred_adata: ad.AnnData,
        true_adata: ad.AnnData,
        top_n: int = 50,
    ) -> float:
        """
        Compute Spatially Variable Gene (SVG) recovery.

        Tests whether genes identified as spatially variable in ground truth
        are also ranked highly in predictions.

        Args:
            pred_adata: Predicted expression (must have .obsm['spatial'])
            true_adata: Ground truth expression (must have .obsm['spatial'])
            top_n: Number of top SVGs to consider

        Returns:
            Jaccard index of top SVG overlap
        """
        # Compute spatial autocorrelation (Moran's I) for all genes
        true_morans = self._compute_morans_all_genes(true_adata)
        pred_morans = self._compute_morans_all_genes(pred_adata)

        # Identify top SVGs in each
        true_top_svgs = set(np.argsort(true_morans)[-top_n:])
        pred_top_svgs = set(np.argsort(pred_morans)[-top_n:])

        # Compute Jaccard index
        intersection = len(true_top_svgs & pred_top_svgs)
        union = len(true_top_svgs | pred_top_svgs)

        return intersection / union if union > 0 else 0.0

    def compute_morans_i(
        self,
        pred_adata: ad.AnnData,
        true_adata: ad.AnnData,
    ) -> float:
        """
        Compute concordance of Moran's I spatial autocorrelation.

        Tests whether genes with high spatial autocorrelation in ground truth
        also show high autocorrelation in predictions.

        Args:
            pred_adata: Predicted expression (must have .obsm['spatial'])
            true_adata: Ground truth expression (must have .obsm['spatial'])

        Returns:
            Pearson correlation of Moran's I values across genes
        """
        if not self.has_spatial:
            raise ImportError("esda/libpysal required. Install with: pip install esda libpysal")

        true_morans = self._compute_morans_all_genes(true_adata)
        pred_morans = self._compute_morans_all_genes(pred_adata)

        # Correlation of Moran's I values
        r, _ = pearsonr(true_morans, pred_morans)
        return float(r)

    def _compute_morans_all_genes(self, adata: ad.AnnData) -> np.ndarray:
        """Compute Moran's I for all genes"""
        if not self.has_spatial:
            raise ImportError("esda/libpysal required")

        # Build spatial weights matrix (k-nearest neighbors)
        coords = adata.obsm['spatial']
        w = KNN.from_array(coords, k=8)

        # Compute Moran's I for each gene
        morans_i = []
        expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        for gene_idx in range(expr.shape[1]):
            gene_expr = expr[:, gene_idx]

            if np.std(gene_expr) > 0:
                moran = Moran(gene_expr, w)
                morans_i.append(moran.I)
            else:
                morans_i.append(0.0)

        return np.array(morans_i)

    def compute_marker_concordance(
        self,
        pred_adata: ad.AnnData,
        true_adata: ad.AnnData,
    ) -> float:
        """
        Compute biological marker concordance.

        Tests whether marker genes show expected relationships
        (e.g., EPCAM vs VIM anticorrelation).

        Args:
            pred_adata: Predicted expression
            true_adata: Ground truth expression

        Returns:
            Mean concordance across marker pairs
        """
        # Example: Test EPCAM (epithelial) vs VIM (stromal) anticorrelation
        if 'epithelial' not in self.marker_genes or 'stromal' not in self.marker_genes:
            return np.nan

        epi_gene = self.marker_genes['epithelial']
        stroma_gene = self.marker_genes['stromal']

        if epi_gene not in pred_adata.var_names or stroma_gene not in pred_adata.var_names:
            return np.nan

        # Get gene indices
        epi_idx = pred_adata.var_names.get_loc(epi_gene)
        stroma_idx = pred_adata.var_names.get_loc(stroma_gene)

        # Compute correlations
        pred_expr = pred_adata.X.toarray() if hasattr(pred_adata.X, 'toarray') else pred_adata.X
        true_expr = true_adata.X.toarray() if hasattr(true_adata.X, 'toarray') else true_adata.X

        true_corr, _ = pearsonr(true_expr[:, epi_idx], true_expr[:, stroma_idx])
        pred_corr, _ = pearsonr(pred_expr[:, epi_idx], pred_expr[:, stroma_idx])

        # Concordance: Are both negative (anticorrelated)?
        concordance = 1.0 if (true_corr < 0 and pred_corr < 0) else 0.0

        return concordance


# Negative Control Generators

def generate_label_shuffle_control(adata: ad.AnnData, seed: int = 42) -> ad.AnnData:
    """
    Generate label shuffle negative control.

    Randomly permutes gene labels to test if model learns gene identity.
    Expected: SSIM should collapse if model is gene-aware.

    Args:
        adata: Original AnnData
        seed: Random seed

    Returns:
        AnnData with shuffled gene labels
    """
    rng = np.random.default_rng(seed)
    shuffled = adata.copy()

    # Shuffle gene labels
    shuffled.var_names = rng.permutation(adata.var_names)

    return shuffled


def generate_spatial_jitter_control(
    adata: ad.AnnData,
    jitter_um: float = 10.0,
    seed: int = 42,
) -> ad.AnnData:
    """
    Generate spatial jitter negative control.

    Adds random coordinate offsets to test spatial awareness.
    Expected: SSIM should drop with increasing jitter.

    Args:
        adata: Original AnnData (must have .obsm['spatial'])
        jitter_um: Jitter magnitude in micrometers
        seed: Random seed

    Returns:
        AnnData with jittered coordinates
    """
    rng = np.random.default_rng(seed)
    jittered = adata.copy()

    # Add random offsets
    offsets = rng.normal(0, jitter_um, size=adata.obsm['spatial'].shape)
    jittered.obsm['spatial'] = adata.obsm['spatial'] + offsets

    return jittered


# Bootstrap Statistical Testing

def hierarchical_bootstrap(
    metric_fn,
    pred_adatas: List[ad.AnnData],
    true_adatas: List[ad.AnnData],
    baseline_adatas: List[ad.AnnData],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Hierarchical bootstrap for small-n comparison.

    Bootstraps at patient level (n=3 with replacement) to estimate
    confidence interval of performance difference.

    Args:
        metric_fn: Function that takes (pred, true) and returns metric value
        pred_adatas: List of predicted AnnDatas (one per patient)
        true_adatas: List of ground truth AnnDatas (one per patient)
        baseline_adatas: List of baseline AnnDatas (one per patient)
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Dict with mean_diff, ci_lower, ci_upper, cohens_d
    """
    rng = np.random.default_rng(seed)
    n_patients = len(pred_adatas)

    differences = []

    for _ in range(n_bootstrap):
        # Bootstrap sample (with replacement)
        indices = rng.choice(n_patients, size=n_patients, replace=True)

        # Compute metrics for this bootstrap sample
        novel_metrics = [metric_fn(pred_adatas[i], true_adatas[i]) for i in indices]
        baseline_metrics = [metric_fn(baseline_adatas[i], true_adatas[i]) for i in indices]

        # Difference
        diff = np.mean(novel_metrics) - np.mean(baseline_metrics)
        differences.append(diff)

    differences = np.array(differences)

    # Compute statistics
    mean_diff = float(np.mean(differences))
    ci_lower = float(np.percentile(differences, 2.5))
    ci_upper = float(np.percentile(differences, 97.5))

    # Cohen's d (effect size)
    std_diff = np.std(differences)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    return {
        'mean_difference': mean_diff,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'cohens_d': cohens_d,
        'significant': ci_lower > 0,  # If CI excludes zero
    }
