# analyze_rollouts.py 
# uv run analyze_rollouts.py --rollouts_file logs\ascii-eos-L4-D768-rollout-test-01661b5f-2ff7-49b1-9ad8-fee77e14bd1c\rollouts_ascii-eos-L4-D768-rollout-test-01661b5f-2ff7-49b1-9ad8-fee77e14bd1c.parquet --ground_truth_file data\txt\TinyStoriesV2-GPT4-train.parquet  --peek --docs_per_sample 3 --cartography_subsets 4
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

from evaluate import POPSScorer, SyntaxAnalyzer, SemanticTrajectoryAnalyzer
# NEW: Import our memory-efficient sampler
from sampler_utils import ParquetSampler

def run_evaluation(docs_a: list, docs_b: list, peek_mode: bool):
    """
    Runs the full evaluation suite on two lists of document strings.
    """
    results = {}
    
    if peek_mode:
        # Log the samples for qualitative review
        with open(PEEK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write("PEEKING at text samples for this evaluation run:\n")
            f.write("\n--- Corpus A Samples ---\n")
            for doc in docs_a: f.write(doc + "\n---\n")
            f.write("\n--- Corpus B Samples ---\n")
            for doc in docs_b: f.write(doc + "\n---\n")

    text_a = "<eos>".join(docs_a)
    text_b = "<eos>".join(docs_b)
    
    # 1. POPS Score (pre-reduced, but we run it on the sample)
    results['pops_a'] = POPS_SCORER.score_text(text_a)
    results['pops_b'] = POPS_SCORER.score_text(text_b)
    # 2. Syntactic Distance
    results['syntax_dist'] = SYNTAX_ANALYZER.analyze_corpora_distance(text_a, text_b)
    # 3. Semantic Distance
    results['semantic_dist'] = SEMANTIC_ANALYZER.analyze_corpora_distance(text_a, text_b)
    return results

def run_cartography(gt_sampler: ParquetSampler, num_subsets: int, docs_per_subset: int, peek_mode: bool):
    """
    Performs 'cartography of the pretrain' by sampling random subsets
    from the ground truth file and comparing them.
    """
    print(f"\n--- Running Cartography on {num_subsets} subsets of ground truth data ---")
    
    # Efficiently sample N subsets of documents without loading the whole file
    subsets = [gt_sampler.get_random_documents(docs_per_subset) for _ in range(num_subsets)]

    cartography_results = []
    for i, j in tqdm(list(combinations(range(num_subsets), 2)), desc="Cartography Pairwise Comparison"):
        # In peek mode, the samples are already small, so we don't need to resample.
        result = run_evaluation(subsets[i], subsets[j], peek_mode)
        cartography_results.append(result)
        
    df = pd.DataFrame(cartography_results)
    baseline_stats = {
        'syntax_dist': {'mean': df['syntax_dist'].mean(), 'std': df['syntax_dist'].std()},
        'semantic_dist': {'mean': df['semantic_dist'].mean(), 'std': df['semantic_dist'].std()},
    }
    print("\n--- Cartography Baseline Results ---")
    print(pd.DataFrame(baseline_stats))
    return baseline_stats

def analyze_rollouts(rollouts_df: pd.DataFrame, gt_sampler: ParquetSampler, docs_per_eval: int, peek_mode: bool):
    """
    Analyzes model rollouts by comparing them to a fresh, random sample of
    ground truth documents at each step.
    """
    print("\n--- Analyzing Model Rollouts vs. Ground Truth ---")
    
    analysis_results = []
    steps = sorted(rollouts_df['step'].unique())

    for step in steps:
        print(f"\n--- Analyzing Step {step} ---")
        # Get the model's generated documents for this step
        model_docs = rollouts_df[rollouts_df['step'] == step]['decoded_text_cleaned'].tolist()
        
        # Get a NEW random sample of ground truth docs for a fair comparison
        ground_truth_docs = gt_sampler.get_random_documents(docs_per_eval)
        
        result = run_evaluation(model_docs, ground_truth_docs, peek_mode)
        result['step'] = step
        analysis_results.append(result)

    df = pd.DataFrame(analysis_results).set_index('step')
    print("\n--- Rollout Analysis Results ---")
    print(df)
    return df

def generate_plots(results_df, baseline_stats, output_file="rollout_analysis.png"):
    """Generates and saves plots of the analysis results using only matplotlib."""
    print(f"\n--- Generating plots to {output_file} ---")
    # CHANGED: We now have 3 metrics to plot
    num_metrics = 3
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 15), sharex=True)
    #sns.set_theme(style="whitegrid") # Use seaborn for styling if available, otherwise defaults to matplotlib

    # --- NEW: 1. POPS Score Plot ---
    ax = axes[0]
    ax.plot(results_df.index, results_df['pops_a'], marker='o', linestyle='-', label='Model POPS (Generated Text)')
    # Use the mean POPS of the sampled ground truth as the baseline
    mean_gt_pops = results_df['pops_b'].mean()
    ax.axhline(mean_gt_pops, color='g', linestyle='--', label=f'Ground Truth Mean POPS ({mean_gt_pops:.2f})')
    ax.set_title("Phonetic & Orthographic Plausibility (POPS) Evolution")
    ax.set_ylabel("POPS Score (lower is better)")
    ax.legend()

    
    # --- 1. Syntactic Distance Plot ---
    metric = 'syntax_dist'
    ax = axes[0]
    # REWRITTEN: Use matplotlib's native plot() function
    ax.plot(results_df.index, results_df[metric], marker='o', linestyle='-', label='Model vs. Ground Truth')
    
    # The rest of the plotting logic is already pure matplotlib
    ax.axhline(baseline_stats[metric]['mean'], color='r', linestyle='--', label='Ground Truth Mean Self-Distance')
    ax.fill_between(results_df.index, 
                    baseline_stats[metric]['mean'] - baseline_stats[metric]['std'],
                    baseline_stats[metric]['mean'] + baseline_stats[metric]['std'],
                    color='r', alpha=0.1, label='Ground Truth 1-std dev')
    ax.set_title("Syntactic Distance (TED) Evolution")
    ax.set_ylabel("Wasserstein Distance (lower is better)")
    ax.legend()
    ax.grid(True)

    # --- 2. Semantic Distance Plot ---
    metric = 'semantic_dist'
    ax = axes[1]
    # REWRITTEN: Use matplotlib's native plot() function
    ax.plot(results_df.index, results_df[metric], marker='o', linestyle='-', label='Model vs. Ground Truth')

    ax.axhline(baseline_stats[metric]['mean'], color='r', linestyle='--', label='Ground Truth Mean Self-Distance')
    ax.fill_between(results_df.index, 
                    baseline_stats[metric]['mean'] - baseline_stats[metric]['std'],
                    baseline_stats[metric]['mean'] + baseline_stats[metric]['std'],
                    color='r', alpha=0.1, label='Ground Truth 1-std dev')
    ax.set_title("Semantic Trajectory Distance Evolution")
    ax.set_ylabel("Wasserstein Distance (lower is better)")
    ax.set_xlabel("Training Step")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print("Plots saved.")

def main():
    #PEEK_LOG_FILE = "peek_log.txt"
    # Initialize the analyzers once to avoid repeated model loading
    print("Initializing analyzers...")
    POPS_SCORER = POPSScorer()
    SYNTAX_ANALYZER = SyntaxAnalyzer()
    SEMANTIC_ANALYZER = SemanticTrajectoryAnalyzer()
    print("Analyzers ready.")

    parser = argparse.ArgumentParser(description="Analyze model rollouts and perform corpus cartography.")
    parser.add_argument("--rollouts_file", type=str)
    parser.add_argument("--ground_truth_file", type=str)
    parser.add_argument("--peek", action="store_true", help="Enable peeking mode for a fast, sampled report.")
    # CHANGED: We now specify the number of docs to sample, not just a flag
    parser.add_argument("--docs_per_sample", type=int, default=3, help="Number of documents per evaluation sample.")
    parser.add_argument("--cartography_subsets", type=int, default=3)
    args = parser.parse_args()

    base_output_path = args.rollouts_file.replace(".parquet", "")
    peek_log_path = f"{base_output_path}_peek_log.txt"
    plot_output_path = f"{base_output_path}_analysis.png"

    if args.peek:
        open(peek_log_path, "w").close() 
        print(f"PEEK MODE ENABLED: Using and logging {args.docs_per_sample} docs per comparison.")

    # NEW: Instantiate our memory-efficient sampler instead of loading everything
    gt_sampler = ParquetSampler(args.ground_truth_file)
    
    # 1. Establish the baseline
    baseline_stats = run_cartography(gt_sampler, args.cartography_subsets, args.docs_per_sample, args.peek)

    # 2. Analyze the model's performance
    rollouts_df = pd.read_parquet(args.rollouts_file)
    results_df = analyze_rollouts(rollouts_df, gt_sampler, args.docs_per_sample, args.peek)

    # 3. Visualize the results
    generate_plots(results_df, baseline_stats, output_file=plot_output_path)

if __name__ == '__main__':

    main()