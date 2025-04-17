import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import random
from collections import defaultdict
import os
from tqdm import tqdm
from datetime import datetime
import warnings
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# ---------------------------------------------
# EigenLayer Simulation
# ---------------------------------------------
# Simulates economic outcomes for EigenLayer operators who restake ETH across AVSs (Actively Validated Services).
# Monte Carlo simulation models economic risks (slashing) and returns (APR) under multiple failure scenarios.
# Outputs include scenario-based CSVs, summary statistics, and high-quality visualizations for analysis/reporting.

# ---------------------------------------------
# SECTION 0: Seed Control for Reproducibility
# ---------------------------------------------
# Allows switching between reproducible results (SEED = 42) or randomized results using current timestamp
USE_RANDOM_SEED = False          # Set to True to use a new seed every time you run, False for fixed seed
SEED = int(datetime.now().timestamp()) if USE_RANDOM_SEED else 42
np.random.seed(SEED)             # Makes NumPy random outputs reproducible
random.seed(SEED)                # Makes Python random outputs reproducible

# ---------------------------------------------
# SECTION 1: Global Parameters and Initialization
# ---------------------------------------------
NUM_OPERATORS = 1000     # Number of simulated operators staking via EigenLayer
NUM_AVS = 100            # Number of Actively Validated Services (AVSs)
ETH_APR = 0.03           # Base Ethereum staking return (APR)
NUM_RUNS = 1000          # Monte Carlo iterations per scenario for robustness
CORRELATION_FACTOR = 0.15  # Increase slashing probability for each additional failed AVS

# ---------------------------------------------
# Print Simulation Overview to Terminal
# ---------------------------------------------
# This provides a readable summary of the simulation parameters before execution,
# useful for understanding the run context, debugging, or replicating results later.
# Includes operator count, AVS count, APR assumption, iteration count, and seed.
print("=" * 60)
print("Running EigenLayer Operator Simulation")
print("- Simulates economic outcomes from restaking ETH across AVSs")
print("- Models operator rewards and slashing risks under 4 scenarios")
print(f"- Monte Carlo simulation with {NUM_OPERATORS} operators × {NUM_RUNS} iterations")
print(f"- {NUM_AVS} Actively Validated Services (AVSs) simulated")
print(f"- Assumes Ethereum base staking APR = {ETH_APR:.2%}")
print(f"- Using correlation factor = {CORRELATION_FACTOR} for recursive slashing")
print(f"- Simulation SEED: {SEED}")
print("=" * 60)
print("=" * 60)

# ---------------------------------------------
# Continue SECTION 1: Global Parameters and Initialization
# ---------------------------------------------
# Four failure scenarios: increasing in severity
SCENARIOS = ["none", "single", "cascade", "panic"]
SCENARIO_LABELS = {
    "none": "No AVS failure – base case",
    "single": "1 AVS failure – isolated incident",
    "cascade": "5 AVS failure – correlated slashing",
    "panic": "20 AVS failure – extreme network event"
}

# Create timestamped directory to store output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Create subfolder for all generated visualizations
visuals_dir = os.path.join(results_dir, "visualizations")
os.makedirs(visuals_dir, exist_ok=True)

# Save configuration parameters for reference and reproducibility
with open(f"{results_dir}/config.txt", "w") as f:
    f.write("=== Simulation Parameters ===\n")
    f.write(f"SEED: {SEED}\n")
    f.write(f"NUM_OPERATORS: {NUM_OPERATORS}\n")
    f.write(f"NUM_AVS: {NUM_AVS}\n")
    f.write(f"ETH_APR: {ETH_APR}\n")
    f.write(f"NUM_RUNS: {NUM_RUNS}\n")
    f.write(f"CORRELATION_FACTOR: {CORRELATION_FACTOR}\n")

# ---------------------------------------------
# SECTION 2: AVS Reward & Slashing Assignments
# ---------------------------------------------
# AVSs are grouped into tiers by risk/reward. Higher reward = higher potential slashing loss.
reward_tiers = {
    "low": (0.0025, 0.004),    # +0.25–0.40% APR, low risk
    "medium": (0.004, 0.007),  # +0.40–0.70% APR, moderate risk
    "high": (0.007, 0.01)      # +0.70–1.00% APR, high risk
}

avs_reward_boosts = {}         # Dict: AVS ID → reward boost
avs_slash_penalties = {}       # Dict: AVS ID → penalty percentage (max 100%)

# Assign each AVS a reward and a proportional slashing penalty
for avs_id in range(NUM_AVS):
    tier = random.choice(list(reward_tiers.keys()))
    reward = round(np.random.uniform(*reward_tiers[tier]), 4)
    penalty = round(reward * np.random.uniform(6, 12), 4)  # Higher APR = higher risk (proportional to reward)
    avs_reward_boosts[avs_id] = reward
    avs_slash_penalties[avs_id] = min(penalty, 1.0)        # Cap penalty at 100% of stake

# ---------------------------------------------
# SECTION 3: Operator Risk Profiles
# ---------------------------------------------
# Defines how many AVSs each operator engages with, based on risk appetite.
# More aggressive = more AVSs = more risk exposure.
RISK_PROFILES = {
    "conservative": {"prob": 0.4, "avs_pct_range": (0.01, 0.05)},
    "moderate": {"prob": 0.35, "avs_pct_range": (0.03, 0.10)},
    "aggressive": {"prob": 0.2, "avs_pct_range": (0.05, 0.25)},
    "ultra_aggressive": {"prob": 0.05, "avs_pct_range": (0.10, 1.00)}
}

# Generate operator population with assigned risk profiles and AVS selections
def generate_operators():
    operators = []
    risk_keys = list(RISK_PROFILES.keys())
    risk_probs = [RISK_PROFILES[k]["prob"] for k in risk_keys]
    for i in range(NUM_OPERATORS):
        profile = np.random.choice(risk_keys, p=risk_probs)
        avs_pct_range = RISK_PROFILES[profile]["avs_pct_range"]
        avs_percent = np.random.uniform(*avs_pct_range)
        num_avs = max(1, int(NUM_AVS * avs_percent))
        assigned_avs = random.sample(range(NUM_AVS), min(num_avs, NUM_AVS))
        operators.append({"id": i, "profile": profile, "avs": assigned_avs})
    return operators

# ---------------------------------------------
# SECTION 4: Core Simulation Logic
# ---------------------------------------------
# Simulates operator earnings or losses depending on scenario and exposure
def simulate_event(scenario, operators):
    # Set of AVSs that fail (based on scenario type)
    if scenario == "none":
        slashed_avss = set()
    elif scenario == "single":
        slashed_avss = {random.choice(range(NUM_AVS))}
    elif scenario == "cascade":
        slashed_avss = set(random.sample(range(NUM_AVS), 5))
    elif scenario == "panic":
        slashed_avss = set(random.sample(range(NUM_AVS), 20))

    earnings_data = []
    for op in operators:
        reward_bonus = sum([avs_reward_boosts[avs] for avs in op["avs"]])
        restaked_apr = ETH_APR + reward_bonus

        # Count how many of operator's AVSs are in the failed set
        failed_avs_count = sum(1 for avs in op["avs"] if avs in slashed_avss)
        
        # Apply slashing penalty if any of operator's AVSs are in the failed set
        if failed_avs_count > 0:
            # Base penalty calculation
            base_penalty = sum([avs_slash_penalties[avs] for avs in op["avs"] if avs in slashed_avss])
            
            # Apply correlation factor for multiple failed AVSs
            if failed_avs_count > 1:
                # Increase penalty by correlation factor for each additional failed AVS
                correlation_multiplier = 1 + (CORRELATION_FACTOR * (failed_avs_count - 1))
                penalty = base_penalty * correlation_multiplier
            else:
                penalty = base_penalty
                
            penalty = min(penalty, 1.0)  # Cap penalty at 100%
            operator_earnings = -penalty  # Operator loses value
            slashed = True
        else:
            operator_earnings = restaked_apr
            slashed = False

        staker_apr = round(operator_earnings * 0.85, 4)  # Assumption: Delegators get 85% of earnings

        earnings_data.append({
            "profile": op["profile"],
            "avs_count": len(op["avs"]),
            "failed_avs_count": failed_avs_count,  # Add this to track correlation effects
            "earnings": round(operator_earnings, 4),
            "delegator_apr": staker_apr,
            "slashed": slashed
        })

    return pd.DataFrame(earnings_data)

# ---------------------------------------------
# SECTION 5: Monte Carlo Execution
# ---------------------------------------------
aggregated_results = {scenario: [] for scenario in SCENARIOS}
summary_stats = []

for scenario in tqdm(SCENARIOS, desc="Running all scenarios"):
    for i in tqdm(range(NUM_RUNS), desc=f"Simulating {scenario}", leave=False):
        ops = generate_operators()
        df = simulate_event(scenario, ops)
        df["iteration"] = i + 1  # Track which run this was from
        aggregated_results[scenario].append(df)

# ---------------------------------------------
# SECTION 6: Metrics and Summary Statistics
# ---------------------------------------------
for scenario in tqdm(SCENARIOS, desc="Processing results"):
    combined = pd.concat(aggregated_results[scenario])
    # Extra safety check to ensure output directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    combined.to_csv(f"{results_dir}/eigenlayer_{scenario}_full_output.csv", index=False)

    stats = {
        "Scenario": scenario,
        "Description": SCENARIO_LABELS[scenario],
        "% Slashed": round(100 * combined["slashed"].mean(), 2),
        "Avg Operator APR": round(combined["earnings"].mean(), 4),
        "Median Operator APR": round(combined["earnings"].median(), 4),
        "Avg Delegator APR": round(combined["delegator_apr"].mean(), 4),
        "Std Dev (Operator APR)": round(combined["earnings"].std(), 4),
        "Min APR": round(combined["earnings"].min(), 4),
        "Max APR": round(combined["earnings"].max(), 4),
        "Q1 APR": round(combined["earnings"].quantile(0.25), 4),
        "Q3 APR": round(combined["earnings"].quantile(0.75), 4)
    }

    # Add per-profile slashing rates to stats
    slashed_by_profile = combined[combined["slashed"]].groupby("profile").size().to_dict()
    total_by_profile = combined.groupby("profile").size().to_dict()
    for profile in total_by_profile:
        pct = 100 * slashed_by_profile.get(profile, 0) / total_by_profile[profile]
        stats[f"% Slashed ({profile})"] = round(pct, 2)

    summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(f"{results_dir}/eigenlayer_summary_statistics.csv", index=False)

# Print summary stats to console for quick review (sorted by % Slashed for clarity)
print("\n=== Summary Statistics ===")
print(summary_df.sort_values(by="% Slashed", ascending=False).to_string(index=False))

# ---------------------------------------------
# SECTION 7: Visualizations 
# ---------------------------------------------
# Define shared color palette to ensure consistency across plots
profile_palette = {
    "conservative": "#1f77b4",  # Blue
    "moderate": "#2ca02c",      # Green
    "aggressive": "#ff7f0e",    # Orange
    "ultra_aggressive": "#d62728"  # Red
}

# Define consistent order for risk profiles
risk_profile_order = ["conservative", "moderate", "aggressive", "ultra_aggressive"]

# Create consistent legends
custom_legend = [
    Patch(color=profile_palette["conservative"], label="Conservative (1-5% of AVSs)"),
    Patch(color=profile_palette["moderate"], label="Moderate (3-10% of AVSs)"),
    Patch(color=profile_palette["aggressive"], label="Aggressive (5-25% of AVSs)"),
    Patch(color=profile_palette["ultra_aggressive"], label="Ultra-Aggressive (10-100% of AVSs)")
]

# Determine global min and max APR for consistent y-axis ranges
global_min_apr = min(df["earnings"].min() for df in aggregated_results.values() for df in df)
global_max_apr = max(df["earnings"].max() for df in aggregated_results.values() for df in df)

# Function for consistent boxplots
def create_consistent_boxplot(scenario, df, output_path):
    plt.figure(figsize=(10, 6))
    
    # Create a new DataFrame with reordered profiles to ensure consistent ordering
    plot_data = []
    for profile in risk_profile_order:
        profile_data = df[df["profile"] == profile]
        if not profile_data.empty:
            plot_data.append(profile_data)
    
    if plot_data:
        plot_df = pd.concat(plot_data)
        
        # Create boxplot with consistent order
        sns.boxplot(x="profile", y="earnings", hue="profile", data=plot_df, 
                    palette=profile_palette, order=risk_profile_order, dodge=False, legend=False)
        
        plt.title(f"APR by Risk Profile – {scenario.capitalize()} Scenario")
        plt.xlabel("Operator Risk Profile")
        plt.ylabel("Operator APR")
        plt.ylim(global_min_apr, global_max_apr)  # Consistent y-axis
        plt.grid(True)
        
        # Place legend outside plot
        plt.legend(handles=custom_legend, title="Operator Risk Profile", 
                  bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

# Function for faceted boxplot grid across scenarios
def create_faceted_apr_boxplots(output_path):
    scenario_order = ["none", "single", "cascade", "panic"]
    scenario_labels = {
        "none": "No AVS Failure (Base Case)",
        "single": "1 AVS Failure (Isolated Incident)",
        "cascade": "5 AVS Failure (Correlated Slashing)",
        "panic": "20 AVS Failure (Extreme Network Event)"
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenario_order):
        ax = axes[idx]
        df = pd.concat(aggregated_results[scenario])
        df["profile"] = pd.Categorical(df["profile"], categories=risk_profile_order, ordered=True)

        sns.boxplot(
            x="profile", y="earnings", hue="profile", data=df, ax=ax,
            order=risk_profile_order, palette=profile_palette, dodge=False, legend=False 
        )
        ax.set_title(scenario_labels[scenario])
        ax.set_xlabel("Risk Profile")
        ax.set_ylabel("Operator APR")
        ax.set_ylim(global_min_apr, global_max_apr)
        ax.grid(True)

    fig.suptitle("Operator APR Distribution by Risk Profile and Scenario", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()

# Generate the faceted APR boxplot across scenarios
create_faceted_apr_boxplots(f"{visuals_dir}/apr_boxplot_faceted_by_scenario.png")


# Function for consistent histograms
def create_consistent_histogram(scenario, df, output_path):
    plt.figure(figsize=(10, 6))
    
    # Create histogram with consistent bins and range
    sns.histplot(df["earnings"], bins=50)
    
    plt.title(f"Operator Earnings Distribution – {scenario.capitalize()} Scenario")
    plt.xlabel("Operator APR")
    plt.ylabel("Number of Operators")
    plt.xlim(global_min_apr, global_max_apr)
    plt.grid(True)
    plt.tight_layout()
    # Extra safety check to ensure visuals directory exists
    if not os.path.exists(visuals_dir):
        os.makedirs(visuals_dir)
    plt.savefig(output_path)
    plt.close()

# Function for facet grid of histograms
def create_facet_histograms(output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Consistent order of scenarios
    scenario_order = ["none", "single", "cascade", "panic"]
    scenario_labels = {
        "none": "No AVS Failure (Base Case)",
        "single": "1 AVS Failure (Isolated Incident)",
        "cascade": "5 AVS Failure (Correlated Slashing)",
        "panic": "20 AVS Failure (Extreme Network Event)"
    }
    
    for idx, scenario in enumerate(scenario_order):
        ax = axes[idx // 2, idx % 2]
        data = pd.concat(aggregated_results[scenario])
        
        sns.histplot(data["earnings"], bins=50, ax=ax)
        
        ax.set_title(f"{scenario_labels[scenario]}")
        ax.set_xlabel("Operator APR")
        ax.set_ylabel("Count")
        ax.set_xlim(global_min_apr, global_max_apr)
        ax.grid(True)
    
    plt.tight_layout()
    # Extra safety check to ensure visuals directory exists
    if not os.path.exists(visuals_dir):
        os.makedirs(visuals_dir)
    plt.savefig(output_path)
    plt.close()

# Generate individual histograms
for scenario in SCENARIOS:
    df = pd.concat(aggregated_results[scenario])
    create_consistent_histogram(scenario, df, f"{visuals_dir}/earnings_histogram_{scenario}.png")

# Generate facet grid of histograms
create_facet_histograms(f"{visuals_dir}/facet_histograms_all_scenarios.png")

# Generate boxplots by risk profile
for scenario in SCENARIOS:
    df = pd.concat(aggregated_results[scenario])
    create_consistent_boxplot(scenario, df, f"{visuals_dir}/apr_by_risk_profile_{scenario}.png")

# Function to create summary statistics table with improved formatting
def plot_summary_table(df, output_path):
    # Rename long column names to prevent overlap
    df = df.rename(columns={
        "Avg Operator APR": "Avg Op APR",
        "Median Operator APR": "Med Op APR",
        "Avg Delegator APR": "Avg Del APR",
        "Std Dev (Operator APR)": "Std Dev APR",
        "% Slashed (aggressive)": "% Slashed (aggr.)",
        "% Slashed (ultra_aggressive)": "% Slashed (ultra)",
        "% Slashed (moderate)": "% Slashed (mod.)",
        "% Slashed (conservative)": "% Slashed (cons.)"
    })

    # Consistent ordering of rows based on scenario severity
    scenario_order = ["none", "single", "cascade", "panic"]
    df = df.set_index("Scenario").loc[scenario_order].reset_index()

    # Create a more readable description column
    description_map = {
        "No AVS failure – base case": "No failures (base case)",
        "1 AVS failure – isolated incident": "Single AVS failure",
        "5 AVS failure – correlated slashing": "5 AVS failures (correlated)",
        "20 AVS failure – extreme network event": "20 AVS failures (extreme)"
    }
    df["Description"] = df["Description"].map(lambda x: description_map.get(x, x))

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.axis("tight")
    ax.axis("off")

    # Create the table with improved formatting
    table = ax.table(
        cellText=df.round(3).values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.4)  # Slightly wider

    # Style improvements for readability
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold', wrap=True)
            cell.set_height(0.15)  # Taller header cells for wrapped text
        if col == 0 or col == 1:  # Scenario and description columns
            cell.set_text_props(ha='left')
            cell.set_width(0.15)

    plt.title("Exhibit: Summary Statistics Across Scenarios", fontsize=14, pad=10)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

# Create summary statistics table
plot_summary_table(summary_df, f"{visuals_dir}/summary_statistics_table.png")

# ---------------------------------------------
# SECTION 7.1: Correlation Factor Impact Analysis - with Improved Data Collection
# ---------------------------------------------
    
def analyze_correlation_impact():
    """Analyze and visualize the impact of the correlation factor on slashing severity"""
    
    print("\n=== Correlation Factor Impact Analysis ===")
    
    # Process ALL scenarios to collect any available data
    correlation_data = []
    
    # Consistent scenario order and colors
    scenario_order = ["single", "cascade", "panic"]
    scenario_colors = {
        "single": "#ff9f1c",  # Light orange
        "cascade": "#2ec4b6",  # Teal
        "panic": "#e71d36"     # Red
    }
    scenario_labels = {
        "single": "1 AVS failure – isolated incident",
        "cascade": "5 AVS failure – correlated slashing",
        "panic": "20 AVS failure – extreme network event"
    }
    
    # First collect all available data points from all scenarios
    for scenario in scenario_order:
        combined = pd.concat(aggregated_results[scenario])
        
        # Look at slashed operators
        slashed_ops = combined[combined["slashed"]]
        
        # Group by number of failed AVSs
        if not slashed_ops.empty:
            by_failed_count = slashed_ops.groupby("failed_avs_count").agg({
                "earnings": ["mean", "min", "count"],
                "profile": "count"
            })
            
            # Format for plotting
            for count, row in by_failed_count.iterrows():
                correlation_data.append({
                    "scenario": scenario,
                    "failed_avs_count": count,
                    "avg_loss": -row[("earnings", "mean")],  # Convert to positive loss value
                    "max_loss": -row[("earnings", "min")],
                    "operator_count": row[("profile", "count")]
                })
    
    if correlation_data:
        corr_df = pd.DataFrame(correlation_data)
        
        # Create the enhanced visualization with explanatory annotation
        plt.figure(figsize=(12, 7))
        
        # Calculate expected penalty based on correlation factor for theoretical curve
        # This will show how losses should scale with the correlation factor
        max_failed_avs = 20
        theoretical_x = np.arange(1, max_failed_avs + 1)
        
        # Base penalty for a single AVS failure
        # Find this from the data if available, otherwise use a reasonable value
        base_penalty = 0.05  # Starting with assumption of 5% loss for 1 AVS
        
        # Find the actual base penalty from each scenario if available
        for scenario in scenario_order:
            scenario_data = corr_df[corr_df["scenario"] == scenario]
            if not scenario_data.empty and 1 in scenario_data["failed_avs_count"].values:
                base_penalty_val = scenario_data[scenario_data["failed_avs_count"] == 1]["avg_loss"].values[0]
                if scenario == "single":  # Use single scenario as benchmark if available
                    base_penalty = base_penalty_val
        
        # Calculate theoretical curve based on correlation factor
        theoretical_y = []
        for n in theoretical_x:
            if n == 1:
                penalty = base_penalty
            else:
                # Apply correlation factor multiplicatively
                penalty = min(base_penalty * (1 + CORRELATION_FACTOR * (n - 1)), 1.0)
            theoretical_y.append(penalty)
        
        # Plot theoretical curve
        plt.plot(theoretical_x, theoretical_y, 'k--', alpha=0.5, 
                 label=f"Theoretical (correlation factor: {CORRELATION_FACTOR})")
        
        # Plot actual data for each scenario
        for scenario in scenario_order:
            scenario_data = corr_df[corr_df["scenario"] == scenario]
            
            # Only plot if there's data (and more than one data point ideally)
            if not scenario_data.empty and len(scenario_data) > 0:
                # Sort by failed_avs_count to ensure proper line connections
                scenario_data = scenario_data.sort_values("failed_avs_count")
                
                plt.plot(scenario_data["failed_avs_count"], 
                         scenario_data["avg_loss"], 
                         marker='o', 
                         linewidth=2,
                         color=scenario_colors[scenario],
                         label=scenario_labels[scenario])
                
                # Add explanation for missing data if needed
                if scenario == "single" and len(scenario_data) <= 1:
                    plt.annotate(
                        "Single failure scenario typically\nonly affects 1 AVS per operator",
                        xy=(2, 0.1), 
                        xytext=(3, 0.2),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
                    )
                
                if scenario == "cascade" and max(scenario_data["failed_avs_count"]) < 5:
                    plt.annotate(
                        "Cascade scenario rarely affects\nmore than a few AVSs per operator",
                        xy=(4, 0.3), 
                        xytext=(6, 0.4),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
                    )
        
        plt.xlabel("Number of Failed AVSs for a Single Operator")
        plt.ylabel("Average Loss (negative APR)")
        plt.title("Impact of Multiple AVS Failures on Slashing Severity")
        plt.grid(True)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(handles=handles, labels=labels)

        
        # Add explanation of correlation factor
        plt.figtext(0.5, 0.01, 
                   f"Correlation Factor ({CORRELATION_FACTOR}): When multiple AVSs fail simultaneously, each additional AVS\n" +
                   f"increases the slashing penalty by {CORRELATION_FACTOR*100}% multiplicatively.",
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the explanation text
        plt.savefig(f"{visuals_dir}/correlation_factor_impact.png")
        plt.close()
        
        # Save the data for reference
        corr_df.to_csv(f"{results_dir}/correlation_impact_analysis.csv", index=False)
        
        # Print summary to terminal for traceability
        print(f"Correlation impact chart saved to: {visuals_dir}/correlation_factor_impact.png")
        print(f"Raw data saved to: {results_dir}/correlation_impact_analysis.csv")
        print(f"Correlation data points collected: {len(corr_df)}")
        print(f"Max observed failed AVSs in one scenario: {corr_df['failed_avs_count'].max()}")

        return corr_df
    else:
        print("No correlation data available for analysis.")
        return None
        
# Call the correlation analysis function
correlation_df = analyze_correlation_impact()

# Operator APR Boxplot by Scenario and Risk Profile
# Combine all scenario DataFrames (no sampling)
full_frames = []
for scenario, data in aggregated_results.items():
    df = pd.concat(data)
    df["scenario"] = scenario.capitalize()
    full_frames.append(df)

combined_full_df = pd.concat(full_frames, ignore_index=True)
combined_full_df["earnings_percent"] = combined_full_df["earnings"] * 100

# Define risk profile order for consistency
profile_order = ["conservative", "moderate", "aggressive", "ultra_aggressive"]

# Generate and save boxplot
plt.figure(figsize=(16, 10))
ax = sns.boxplot(
    data=combined_full_df, 
    x="profile", 
    y="earnings_percent", 
    hue="scenario", 
    order=profile_order
)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Failure Scenario", loc="upper right")

plt.title("Operator APR Distribution by Risk Profile and Scenario")
plt.xlabel("Validator Risk Profile")
plt.ylabel("Annual Percentage Rate (APR %)")
plt.grid(True)
plt.tight_layout()

# Save figure to visualization directory
output_path = os.path.join(visuals_dir, "operator_apr_boxplot_by_scenario_full.png")
# Extra safety check to ensure visuals directory exists
if not os.path.exists(visuals_dir):
    os.makedirs(visuals_dir)
plt.savefig(output_path)
plt.close()

# ---------------------------------------------
# SECTION 7.2 Risk Exposure Analysis - with Improved Risk Tolerance Explanation
# ---------------------------------------------
def analyze_risk_thresholds():
    """Analyze results to determine recommended maximum AVS exposure levels"""
    
    print("\n=== Risk Exposure Analysis ===")
    
    # Create a DataFrame to analyze slashing risk by AVS count
    risk_by_avs_count = {}
    
    for scenario in SCENARIOS:
        combined = pd.concat(aggregated_results[scenario])
        
        # Calculate average slashing rate by AVS count
        avs_group = combined.groupby("avs_count")
        avs_counts = avs_group["slashed"].mean() * 100
        
        # Also calculate average APR by AVS count
        avg_apr = avs_group["earnings"].mean()
        
        # Combine into a DataFrame
        scenario_df = pd.DataFrame({
            "slashing_pct": avs_counts,
            "avg_apr": avg_apr
        })
        
        # Store results
        risk_by_avs_count[scenario] = scenario_df
    
    # Combine all scenarios into one DataFrame for easier analysis
    all_data = []
    
    # Consistent scenario order
    scenario_order = ["none", "single", "cascade", "panic"]
    scenario_colors = {
        "none": "#4d8fac",     # Blue
        "single": "#ff9f1c",   # Light orange
        "cascade": "#2ec4b6",  # Teal
        "panic": "#e71d36"     # Red
    }
    scenario_labels = {
        "none": "No AVS failure – base case",
        "single": "1 AVS failure – isolated incident",
        "cascade": "5 AVS failure – correlated slashing",
        "panic": "20 AVS failure – extreme network event"
    }
    
    for scenario in scenario_order:
        df = risk_by_avs_count[scenario]
        for avs_count, row in df.iterrows():
            all_data.append({
                "scenario": scenario,
                "avs_count": avs_count,
                "slashing_pct": row["slashing_pct"],
                "avg_apr": row["avg_apr"]
            })
    
    risk_analysis_df = pd.DataFrame(all_data)
    risk_analysis_df.to_csv(f"{results_dir}/risk_by_avs_count.csv", index=False)
    
    # Find thresholds where risk becomes unacceptable
    # Define risk tolerance levels with descriptions
    risk_tolerance = {
        "low": {"value": 5.0, "description": "Conservative validators (5% max slashing probability)"},
        "medium": {"value": 15.0, "description": "Moderate risk tolerance (15% max slashing probability)"},
        "high": {"value": 30.0, "description": "Aggressive risk appetite (30% max slashing probability)"}
    }
    
    thresholds = {}
    
    # For each scenario (except "none"), find max AVS count for each risk tolerance
    for scenario in ['single', 'cascade', 'panic']:
        thresholds[scenario] = {}
        scenario_data = risk_analysis_df[risk_analysis_df["scenario"] == scenario]
        
        for tolerance, info in risk_tolerance.items():
            max_pct = info["value"]
            # Find maximum AVS count where risk is below tolerance threshold
            below_threshold = scenario_data[scenario_data["slashing_pct"] <= max_pct]
            if not below_threshold.empty:
                max_safe_count = below_threshold["avs_count"].max()
            else:
                max_safe_count = 0
            thresholds[scenario][tolerance] = max_safe_count
    
    # Print recommendations
    print("\nRecommended Maximum AVS Exposure Guidelines:")
    for tolerance, info in risk_tolerance.items():
        print(f"\n{info['description']}:")
        for scenario in ['single', 'cascade', 'panic']:
            print(f"  - {scenario_labels[scenario]}: ≤ {thresholds[scenario][tolerance]} AVSs")
    
    # Create visualization of risk vs. AVS count for different scenarios
    plt.figure(figsize=(12, 7))
    
    # Plot each scenario line
    for scenario in scenario_order:
        scenario_data = risk_analysis_df[risk_analysis_df["scenario"] == scenario]
        plt.plot(scenario_data["avs_count"], 
                 scenario_data["slashing_pct"], 
                 marker='o', 
                 linewidth=2,
                 color=scenario_colors[scenario],
                 label=scenario_labels[scenario])
    
    # Add horizontal lines for risk tolerance levels with improved visuals
    tolerance_colors = {
        "low": "#a2d5f2",      # Light blue
        "medium": "#07689f",   # Medium blue
        "high": "#0a1931"      # Dark blue
    }
    
    # Add risk tolerance horizontal lines with clearer labels
    for tolerance, info in risk_tolerance.items():
        plt.axhline(y=info["value"], 
                  linestyle='--', 
                  alpha=0.7,
                  color=tolerance_colors[tolerance],
                  linewidth=2)
        # Add annotations directly on the lines for better clarity
        plt.text(95, info["value"] + 1, 
                info["description"],
                color=tolerance_colors[tolerance],
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.xlabel('Number of AVSs Validated')
    plt.ylabel('Probability of Getting Slashed (%)')
    plt.title('Slashing Risk by Number of AVSs Validated')
    plt.grid(True)
    plt.legend(loc='upper left')
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
               "Horizontal lines represent different risk tolerance levels. The intersection points with scenario lines\n" +
               "indicate the maximum number of AVSs that can be validated within each risk tolerance.", 
               ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the explanation text
    plt.savefig(f"{visuals_dir}/risk_exposure_guidelines.png")
    plt.close()
    
    # Risk-Return Analysis: Plot APR vs Risk - IMPROVED
    plt.figure(figsize=(12, 7))
    
    # Plot each scenario with improved visibility
    for scenario in scenario_order:
        scenario_data = risk_analysis_df[risk_analysis_df["scenario"] == scenario]
        plt.scatter(scenario_data["slashing_pct"], 
                  scenario_data["avg_apr"], 
                  s=80,
                  alpha=0.7,
                  color=scenario_colors[scenario],
                  label=scenario_labels[scenario])
    
    # Annotate strategic AVS counts for clarity
    key_avs_counts = [5, 10, 25, 50, 75, 100]

    for scenario in scenario_order:
        scenario_data = risk_analysis_df[risk_analysis_df["scenario"] == scenario]
        for _, row in scenario_data.iterrows():
            if row["avs_count"] in key_avs_counts:
                plt.annotate(f"{int(row['avs_count'])} AVSs",
                    (row["slashing_pct"], row["avg_apr"]),
                     xytext=(5, 5),
                     textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )

        plt.axhline(y=0, linestyle='-', color='black', alpha=0.3)
        plt.xlabel('Risk (Probability of Getting Slashed %)')
        plt.ylabel('Return (Average Operator APR)')
        plt.title('Risk-Return Profile by AVS Count and Scenario')
        plt.grid(True)

    # Handle legend more safely to avoid warnings
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(handles=handles, labels=labels)

    # Add explanation text
    plt.figtext(0.5, 0.01, 
            "This chart shows the risk-return tradeoff for different AVS counts. Points above the horizontal line (y=0)\n"
            "represent positive expected returns, while points below indicate expected losses.",
            ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the explanation text
    plt.savefig(f"{visuals_dir}/risk_return_profile.png")
    plt.close()

    return thresholds, risk_analysis_df


# Call the risk analysis function
thresholds, risk_analysis_df = analyze_risk_thresholds()

# -------------------------------------------------
# END OF SIMULATION CODE
# -------------------------------------------------
# Show location of saved results so user can quickly find them
print(f"\n All simulation results saved to: {results_dir}")

# Close any remaining figures 
plt.close('all')

# Confirm completion in terminal
print("Simulation Complete") 

# -------------------------------------------------
# END OF FILE