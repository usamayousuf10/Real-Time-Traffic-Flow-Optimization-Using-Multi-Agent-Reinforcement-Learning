#!/usr/bin/env python3
"""
Analysis and Visualization of PressLight Training
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_training_log(log_path='logs/training_log.json'):
    """Analyze training results"""
    
    try:
        with open(log_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_path}")
        print("   Run training first!")
        return None
    
    episodes = [d['episode'] for d in data]
    queues = [d['queue'] for d in data]
    rewards = [d['reward'] for d in data]
    losses = [d['loss'] for d in data]
    epsilons = [d['epsilon'] for d in data]
    
    print("\n" + "="*70)
    print("TRAINING ANALYSIS")
    print("="*70)
    
    # Summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Total episodes: {len(episodes)}")
    print(f"   Initial queue: {queues[0]:.2f}")
    print(f"   Final queue: {queues[-1]:.2f}")
    print(f"   Best queue: {min(queues):.2f}")
    print(f"   Improvement: {(queues[0] - queues[-1]) / queues[0] * 100:.1f}%")
    
    print(f"\n   Initial epsilon: {epsilons[0]:.3f}")
    print(f"   Final epsilon: {epsilons[-1]:.3f}")
    
    print(f"\n   Initial loss: {losses[0]:.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")
    
    # Phase analysis
    n_episodes = len(episodes)
    phase1_end = n_episodes // 3
    phase2_end = 2 * n_episodes // 3
    
    print(f"\n📈 Phase-wise Performance:")
    print(f"   Phase 1 (0-{phase1_end}): Avg queue = {np.mean(queues[:phase1_end]):.2f}")
    print(f"   Phase 2 ({phase1_end}-{phase2_end}): Avg queue = {np.mean(queues[phase1_end:phase2_end]):.2f}")
    print(f"   Phase 3 ({phase2_end}-{n_episodes}): Avg queue = {np.mean(queues[phase2_end:]):.2f}")
    
    # Expected vs Actual
    print(f"\n🎯 Performance Comparison:")
    print(f"   Your result: {queues[-1]:.2f} queue length")
    print(f"   Target (from research): ~5-6 queue length")
    
    if queues[-1] < 7:
        print(f"   ✅ EXCELLENT! Within target range")
    elif queues[-1] < 9:
        print(f"   ✓ GOOD! Close to target")
    else:
        print(f"   ⚠️  Needs more training or tuning")
    
    # Convergence check
    last_50 = queues[-50:] if len(queues) >= 50 else queues
    variance = np.var(last_50)
    print(f"\n📉 Convergence Analysis:")
    print(f"   Last 50 episodes variance: {variance:.4f}")
    if variance < 0.5:
        print(f"   ✅ CONVERGED - Training is stable")
    elif variance < 1.0:
        print(f"   ✓ CONVERGING - Nearly stable")
    else:
        print(f"   ⚠️  NOT CONVERGED - Continue training")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(episodes, queues, 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Queue Length')
    axes[0, 0].set_title('Queue Length over Training')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=6, color='g', linestyle='--', label='Target (~6)')
    axes[0, 0].legend()
    
    axes[0, 1].plot(episodes, epsilons, 'r-', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].set_title('Epsilon Decay')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.05, color='g', linestyle='--', label='Min (0.05)')
    axes[0, 1].legend()
    
    axes[1, 0].plot(episodes, losses, 'g-', alpha=0.7)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(episodes, rewards, 'm-', alpha=0.7)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Average Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: training_results.png")
    
    print("="*70 + "\n")
    
    return data

def create_comparison_table():
    """Create comparison table with expected results"""
    
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE COMPARISON")
    print("="*70)
    
    print("\n📊 Original Implementation (BEFORE fixes):")
    print("─" * 70)
    print(f"{'Method':<20} {'Travel Time':<15} {'Queue':<10} {'Status':<15}")
    print("─" * 70)
    print(f"{'Fixed-Time':<20} {'631.8s':<15} {'12.64':<10} {'Baseline':<15}")
    print(f"{'Max-Pressure':<20} {'590.2s':<15} {'11.80':<10} {'+6.6%':<15}")
    print(f"{'PressLight (old)':<20} {'790.6s':<15} {'15.81':<10} {'❌ -25.1%':<15}")
    
    print("\n📊 Optimized Implementation (AFTER fixes):")
    print("─" * 70)
    print(f"{'Method':<20} {'Travel Time':<15} {'Queue':<10} {'Status':<15}")
    print("─" * 70)
    print(f"{'Fixed-Time':<20} {'~200s':<15} {'~12.0':<10} {'Baseline':<15}")
    print(f"{'Max-Pressure':<20} {'~150s':<15} {'~9.0':<10} {'+25%':<15}")
    print(f"{'PressLight (new)':<20} {'~100-110s':<15} {'~5-6':<10} {'✅ +45-50%':<15}")
    
    print("\n🔑 Key Improvements Applied:")
    print("─" * 70)
    improvements = [
        ("CityFlow Integration", "Realistic simulation", "vs discrete 5s steps"),
        ("Epsilon Decay", "0.9997 (slow)", "vs 0.995 (too fast)"),
        ("Training Episodes", "500 episodes", "vs 100 episodes"),
        ("Learning Rate", "5e-4", "vs 1e-3"),
        ("Batch Size", "64", "vs 32"),
        ("Buffer Size", "100K", "vs 10K"),
        ("Hidden Dimension", "256", "vs 128"),
        ("Advanced Techniques", "Double DQN + Dueling", "vs basic DQN")
    ]
    
    for name, after, before in improvements:
        print(f"   ✓ {name:<25} {after:<20} {before:<20}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    create_comparison_table()
    analyze_training_log()
