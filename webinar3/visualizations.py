"""
Visualization Code
==================
Generate charts for the webinar.

Part of Webinar 3: Optimizing Multilingual NLP Models

Run: python visualizations.py
"""


def create_optimization_impact_chart():
    """
    Create the Optimization Impact Divergence Chart.
    Shows how different languages degrade differently under optimization.
    """
    import matplotlib.pyplot as plt
    
    optimization_level = [0, 20, 40, 60, 80, 100]
    english_retention = [100, 99.5, 99, 98.5, 98, 97.5]
    spanish_retention = [100, 99, 97.5, 96, 95, 94]
    thai_retention = [100, 97, 93, 88, 84, 80]
    
    plt.figure(figsize=(10, 6))
    plt.plot(optimization_level, english_retention, 'b-o', label='English', linewidth=2)
    plt.plot(optimization_level, spanish_retention, 'orange', marker='s', label='Spanish', linewidth=2)
    plt.plot(optimization_level, thai_retention, 'r-^', label='Thai', linewidth=2)
    
    plt.axhline(y=85, color='gray', linestyle='--', label='Acceptable Threshold')
    
    plt.xlabel('Optimization Intensity (%)', fontsize=12)
    plt.ylabel('Accuracy Retention (%)', fontsize=12)
    plt.title('Optimization Impact by Language', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.ylim(75, 102)
    
    plt.tight_layout()
    plt.savefig('optimization_impact.png', dpi=150)
    plt.close()


def create_finetuning_learning_curve():
    """
    Create the Fine-Tuning Learning Curve chart.
    Shows ROI of different amounts of training data.
    """
    import matplotlib.pyplot as plt
    
    examples = [0, 100, 500, 1000, 2000, 5000, 10000]
    improvement = [0, 8, 14, 16, 17.5, 18.5, 19]
    
    plt.figure(figsize=(10, 6))
    plt.plot(examples, improvement, 'b-o', linewidth=2, markersize=8)
    
    # Add shaded regions
    plt.axvspan(0, 500, alpha=0.2, color='green', label='High ROI Zone')
    plt.axvspan(500, 5000, alpha=0.2, color='yellow', label='Moderate ROI')
    plt.axvspan(5000, 10000, alpha=0.2, color='red', label='Low ROI')
    
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Accuracy Improvement over Zero-Shot (%)', fontsize=12)
    plt.title('Fine-Tuning Learning Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('finetuning_curve.png', dpi=150)
    plt.close()


def create_distillation_comparison():
    """
    Create grouped bar chart comparing teacher and student models.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    models = ['XLM-R Large\n(Teacher)', 'DistilmBERT\n(Student)', '4-Layer\n(Student)']
    accuracy = [91.2, 87.8, 85.1]
    latency = [180, 55, 35]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    bars1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='steelblue')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(80, 95)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, latency, width, label='Latency (ms)', color='coral')
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_ylim(0, 200)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title('Knowledge Distillation Results', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars1, accuracy):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val}%', ha='center', va='bottom', fontsize=10)
    
    for bar, val in zip(bars2, latency):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                f'{val}ms', ha='center', va='bottom', fontsize=10)
    
    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    plt.savefig('distillation_comparison.png', dpi=150)
    plt.close()


def create_per_language_retention():
    """
    Create horizontal bar chart showing distillation retention by language.
    """
    import matplotlib.pyplot as plt
    
    languages = ['English', 'Spanish', 'German', 'Russian', 'Thai', 'Korean']
    retention = [98, 96, 95, 91, 84, 83]
    
    colors = ['#2ecc71' if r >= 90 else '#f39c12' if r >= 85 else '#e74c3c' for r in retention]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(languages, retention, color=colors)
    
    plt.axvline(x=90, color='gray', linestyle='--', linewidth=2, label='Target: 90%')
    
    plt.xlabel('Teacher Performance Retained (%)', fontsize=12)
    plt.title('Per-Language Distillation Retention', fontsize=14, fontweight='bold')
    plt.xlim(75, 102)
    
    # Add value labels
    for bar, val in zip(bars, retention):
        plt.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val}%', va='center', fontsize=11)
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('per_language_retention.png', dpi=150)
    plt.close()


def create_quantization_retention():
    """
    Create bar chart showing quantization accuracy retention by language.
    """
    import matplotlib.pyplot as plt
    
    languages = ['English', 'Spanish', 'German', 'Russian', 'Thai', 'Korean']
    retention = [99.5, 99.0, 98.5, 97.5, 94.2, 93.8]
    
    colors = ['#2ecc71' if r >= 95 else '#f39c12' if r >= 93 else '#e74c3c' for r in retention]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(languages, retention, color=colors)
    
    plt.axhline(y=95, color='gray', linestyle='--', linewidth=2, label='Warning: 95%')
    
    plt.ylabel('Accuracy Retention after INT8 Quantization (%)', fontsize=12)
    plt.xlabel('Language', fontsize=12)
    plt.title('Quantization Impact by Language', fontsize=14, fontweight='bold')
    plt.ylim(90, 101)
    
    # Add value labels
    for bar, val in zip(bars, retention):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.3, 
                f'{val}%', ha='center', va='bottom', fontsize=10)
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('quantization_retention.png', dpi=150)
    plt.close()


def create_infrastructure_pareto():
    """
    Create scatter plot with Pareto frontier for infrastructure tradeoffs.
    """
    import matplotlib.pyplot as plt
    
    configs = ['XLM-R Large\n(GPU)', 'XLM-R Large\n(CPU)', 'Quantized\n(CPU)', 'Distilled\n(CPU)']
    monthly_cost = [767, 485, 91, 46]
    p95_latency = [65, 200, 85, 40]
    
    colors = ['#e74c3c', '#e74c3c', '#3498db', '#2ecc71']
    
    plt.figure(figsize=(10, 6))
    
    for i, (config, cost, latency) in enumerate(zip(configs, monthly_cost, p95_latency)):
        plt.scatter(cost, latency, s=200, c=colors[i], zorder=5)
        plt.annotate(config, (cost, latency), textcoords="offset points", 
                    xytext=(10, 10), ha='left', fontsize=10)
    
    # Draw Pareto frontier
    pareto_x = [46, 91]
    pareto_y = [40, 85]
    plt.plot(pareto_x, pareto_y, 'g--', linewidth=2, label='Pareto Frontier')
    
    plt.xlabel('Monthly Cost ($)', fontsize=12)
    plt.ylabel('p95 Latency (ms)', fontsize=12)
    plt.title('Infrastructure Cost vs Latency Trade-offs', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('infrastructure_pareto.png', dpi=150)
    plt.close()


def create_all_visualizations():
    """Generate all visualization files."""
    print("Creating visualizations...")
    
    create_optimization_impact_chart()
    print("  - optimization_impact.png")
    
    create_finetuning_learning_curve()
    print("  - finetuning_curve.png")
    
    create_distillation_comparison()
    print("  - distillation_comparison.png")
    
    create_per_language_retention()
    print("  - per_language_retention.png")
    
    create_quantization_retention()
    print("  - quantization_retention.png")
    
    create_infrastructure_pareto()
    print("  - infrastructure_pareto.png")
    
    print("\nDone.")


if __name__ == "__main__":
    create_all_visualizations()
