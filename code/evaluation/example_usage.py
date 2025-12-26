"""
Example Usage: SpatialEvaluator for 2μm ST Prediction

Demonstrates how to use the evaluation harness to compare
predicted vs ground truth spatial transcriptomics.

Author: Max Van Belkum
Date: 2025-12-26
"""

import numpy as np
import anndata as ad
from spatial_evaluator import (
    SpatialEvaluator,
    generate_label_shuffle_control,
    generate_spatial_jitter_control,
    hierarchical_bootstrap,
)


def create_dummy_data(n_cells=100, n_genes=50):
    """Create dummy AnnData for testing"""

    # Random expression matrix
    expr = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(float)

    # Spatial coordinates (grid)
    grid_size = int(np.sqrt(n_cells))
    x_coords = np.repeat(np.arange(grid_size), grid_size)[:n_cells]
    y_coords = np.tile(np.arange(grid_size), grid_size)[:n_cells]
    spatial_coords = np.column_stack([x_coords, y_coords]) * 2.0  # 2μm spacing

    # Gene names
    gene_names = [f"Gene_{i}" for i in range(n_genes)]

    # Cell types
    cell_types = np.random.choice(['epithelial', 'immune', 'stromal'], size=n_cells)

    # Create AnnData
    adata = ad.AnnData(
        X=expr,
        obs={'cell_type': cell_types},
        var={'gene_names': gene_names},
    )
    adata.obsm['spatial'] = spatial_coords

    # Add gene categories
    adata.var['category'] = ['epithelial'] * 15 + ['immune'] * 15 + ['stromal'] * 10 + ['housekeeping'] * 10

    return adata


def main():
    """Example evaluation workflow"""

    print("=== SpatialEvaluator Example Usage ===\n")

    # 1. Create dummy data (replace with real data loading)
    print("1. Creating dummy data...")
    true_adata = create_dummy_data(n_cells=144, n_genes=50)  # 12x12 grid
    pred_adata = true_adata.copy()

    # Add some noise to predictions
    noise = np.random.normal(0, 0.5, pred_adata.shape)
    pred_adata.X = pred_adata.X + noise
    pred_adata.X = np.maximum(0, pred_adata.X)  # No negative expression

    # 2. Initialize evaluator with gene categories
    print("2. Initializing evaluator...")
    gene_categories = {
        'epithelial': [g for g, c in zip(true_adata.var_names, true_adata.var['category']) if c == 'epithelial'],
        'immune': [g for g, c in zip(true_adata.var_names, true_adata.var['category']) if c == 'immune'],
        'stromal': [g for g, c in zip(true_adata.var_names, true_adata.var['category']) if c == 'stromal'],
        'housekeeping': [g for g, c in zip(true_adata.var_names, true_adata.var['category']) if c == 'housekeeping'],
    }

    marker_genes = {
        'epithelial': 'Gene_0',  # Dummy marker
        'stromal': 'Gene_15',    # Dummy marker
    }

    evaluator = SpatialEvaluator(
        gene_categories=gene_categories,
        marker_genes=marker_genes,
    )

    # 3. Run comprehensive evaluation
    print("3. Running comprehensive evaluation...")
    results = evaluator.evaluate(
        pred_adata=pred_adata,
        true_adata=true_adata,
        compute_optimal_transport=True,
        compute_bio_conservation=True,
        compute_spatial_fidelity=True,
    )

    print(results.summary())
    print("\n")

    # 4. Test negative controls
    print("4. Testing negative controls...")

    # Label shuffle control
    print("  - Label shuffle control...")
    shuffled_adata = generate_label_shuffle_control(pred_adata, seed=42)
    shuffled_results = evaluator.evaluate(
        pred_adata=shuffled_adata,
        true_adata=true_adata,
        compute_optimal_transport=False,
        compute_bio_conservation=False,
        compute_spatial_fidelity=False,
    )
    print(f"    Original SSIM: {results.ssim_mean:.4f}")
    print(f"    Shuffled SSIM: {shuffled_results.ssim_mean:.4f} (should be much lower)")
    print()

    # Spatial jitter control
    print("  - Spatial jitter control...")
    for jitter_um in [5, 10, 20]:
        jittered_adata = generate_spatial_jitter_control(pred_adata, jitter_um=jitter_um, seed=42)
        jittered_results = evaluator.evaluate(
            pred_adata=jittered_adata,
            true_adata=true_adata,
            compute_optimal_transport=False,
            compute_bio_conservation=False,
            compute_spatial_fidelity=True,
        )
        print(f"    Jitter {jitter_um}μm: SSIM = {jittered_results.ssim_mean:.4f}")
    print()

    # 5. Hierarchical bootstrap (requires multiple patients)
    print("5. Hierarchical bootstrap example...")
    print("  (Simulating 3-patient comparison)")

    # Create 3 "patients"
    pred_patients = [create_dummy_data(n_cells=144, n_genes=50) for _ in range(3)]
    true_patients = [create_dummy_data(n_cells=144, n_genes=50) for _ in range(3)]
    baseline_patients = [create_dummy_data(n_cells=144, n_genes=50) for _ in range(3)]

    # Add noise to predictions
    for i in range(3):
        noise = np.random.normal(0, 0.3 + i*0.1, pred_patients[i].shape)  # Varying noise
        pred_patients[i].X = pred_patients[i].X + noise
        pred_patients[i].X = np.maximum(0, pred_patients[i].X)

        noise_baseline = np.random.normal(0, 0.5, baseline_patients[i].shape)
        baseline_patients[i].X = baseline_patients[i].X + noise_baseline
        baseline_patients[i].X = np.maximum(0, baseline_patients[i].X)

    # Define metric function
    def ssim_metric(pred, true):
        ev = SpatialEvaluator()
        res = ev.evaluate(pred, true, compute_optimal_transport=False,
                          compute_bio_conservation=False, compute_spatial_fidelity=False)
        return res.ssim_mean

    # Run bootstrap
    bootstrap_results = hierarchical_bootstrap(
        metric_fn=ssim_metric,
        pred_adatas=pred_patients,
        true_adatas=true_patients,
        baseline_adatas=baseline_patients,
        n_bootstrap=1000,  # Use 10000 for real analysis
        seed=42,
    )

    print(f"  Mean difference: {bootstrap_results['mean_difference']:.4f}")
    print(f"  95% CI: [{bootstrap_results['ci_95_lower']:.4f}, {bootstrap_results['ci_95_upper']:.4f}]")
    print(f"  Cohen's d: {bootstrap_results['cohens_d']:.4f}")
    print(f"  Significant: {bootstrap_results['significant']}")
    print()

    # 6. Export results
    print("6. Exporting results...")
    results_dict = results.to_dict()
    print(f"  Results dictionary has {len(results_dict)} metrics")

    # Could save to CSV/JSON here
    # pd.DataFrame([results_dict]).to_csv('evaluation_results.csv', index=False)

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
