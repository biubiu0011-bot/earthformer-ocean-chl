#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Data Quality Check
Check training and testing data for anomalies and distribution differences
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Data paths
INPUT_DIR = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'pre_train_data/input/'
OUTPUT_DIR = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'pre_train_data/output/'
TRAIN_INDICES_PATH = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'pre_train_data/valid_train_indices.npy'
TEST_INDICES_PATH = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'pre_train_data/valid_test_indices.npy'

class DataQualityChecker:
    def __init__(self):
        self.train_indices = None
        self.test_indices = None
        self.train_stats = []
        self.test_stats = []
        
    def load_indices(self):
        """Load train and test indices"""
        print("="*80)
        print("Step 1: Loading Data Indices")
        print("="*80)
        
        if os.path.exists(TRAIN_INDICES_PATH):
            self.train_indices = np.load(TRAIN_INDICES_PATH)
            print(f"✓ Train indices loaded: {len(self.train_indices)} samples")
        else:
            self.train_indices = np.arange(1827, 7671, 1)
            print(f"⚠ Using default train range: {len(self.train_indices)} samples")
        
        if os.path.exists(TEST_INDICES_PATH):
            self.test_indices = np.load(TEST_INDICES_PATH)
            print(f"✓ Test indices loaded: {len(self.test_indices)} samples")
        else:
            self.test_indices = np.arange(7671, 9119, 1)
            print(f"⚠ Using default test range: {len(self.test_indices)} samples")
        
        print(f"\nTrain index range: {self.train_indices.min()} - {self.train_indices.max()}")
        print(f"Test index range: {self.test_indices.min()} - {self.test_indices.max()}")
        
    def check_sample(self, idx, is_train=True):
        """Check a single sample for anomalies"""
        try:
            input_path = os.path.join(INPUT_DIR, f'input_{idx}.npy')
            output_path = os.path.join(OUTPUT_DIR, f'output_{idx}.npy')
            
            if not os.path.exists(input_path) or not os.path.exists(output_path):
                return None
            
            input_data = np.load(input_path)
            output_data = np.load(output_path)
            
            
            # Calculate statistics
            stats = {
                'index': idx,
                'is_train': is_train,
                # Input statistics
                'input_mean': np.nanmean(input_data),
                'input_std': np.nanstd(input_data),
                'input_min': np.nanmin(input_data),
                'input_max': np.nanmax(input_data),
                'input_nan_count': np.isnan(input_data).sum(),
                'input_inf_count': np.isinf(input_data).sum(),
                # Output statistics
                'output_mean': np.nanmean(output_data),
                'output_std': np.nanstd(output_data),
                'output_min': np.nanmin(output_data),
                'output_max': np.nanmax(output_data),
                'output_nan_count': np.isnan(output_data).sum(),
                'output_inf_count': np.isinf(output_data).sum(),
                # Combined statistics
                'output_range': np.nanmax(output_data) - np.nanmin(output_data),
                'output_temporal_var': np.var(output_data, axis=0).mean(),
            }
            
            return stats
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None
    
    def scan_dataset(self, indices, is_train=True, sample_rate=1.0):
        """Scan entire dataset"""
        label = "Training" if is_train else "Testing"
        print(f"\n{'='*80}")
        print(f"Step 2: Scanning {label} Dataset")
        print(f"{'='*80}")
        
        # Sample indices if needed
        if sample_rate < 1.0:
            n_samples = int(len(indices) * sample_rate)
            sampled_indices = np.random.choice(indices, n_samples, replace=False)
            print(f"Sampling {n_samples}/{len(indices)} samples ({sample_rate*100:.0f}%)")
        else:
            sampled_indices = indices
        
        stats_list = []
        failed_count = 0
        
        for idx in tqdm(sampled_indices, desc=f"Scanning {label}"):
            stats = self.check_sample(idx, is_train)
            if stats is not None:
                stats_list.append(stats)
            else:
                failed_count += 1
        
        print(f"\n✓ Successfully loaded: {len(stats_list)} samples")
        if failed_count > 0:
            print(f"⚠ Failed to load: {failed_count} samples")
        
        return stats_list
    
    def analyze_statistics(self):
        """Analyze and compare train vs test statistics"""
        print(f"\n{'='*80}")
        print("Step 3: Statistical Analysis")
        print(f"{'='*80}")
        
        train_stats = np.array(self.train_stats)
        test_stats = np.array(self.test_stats)
        
        metrics = [
            ('output_mean', 'Output Mean'),
            ('output_std', 'Output Std'),
            ('output_min', 'Output Min'),
            ('output_max', 'Output Max'),
            ('output_range', 'Output Range'),
            ('output_temporal_var', 'Temporal Variance'),
        ]
        
        print(f"\n{'Metric':<25} {'Train':<15} {'Test':<15} {'Difference':<15}")
        print("-"*70)
        
        for key, label in metrics:
            train_vals = [s[key] for s in self.train_stats]
            test_vals = [s[key] for s in self.test_stats]
            
            train_mean = np.mean(train_vals)
            test_mean = np.mean(test_vals)
            diff_pct = ((test_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0
            
            print(f"{label:<25} {train_mean:<15.4f} {test_mean:<15.4f} {diff_pct:+.2f}%")
        
        # Check for anomalies
        print(f"\n{'='*80}")
        print("Step 4: Anomaly Detection")
        print(f"{'='*80}")
        
        self.detect_anomalies(self.train_stats, "Training")
        self.detect_anomalies(self.test_stats, "Testing")
    
    def detect_anomalies(self, stats_list, label):
        """Detect anomalous samples"""
        print(f"\n{label} Set Anomalies:")
        print("-"*40)
        
        # Extract key metrics
        output_maxs = np.array([s['output_max'] for s in stats_list])
        output_ranges = np.array([s['output_range'] for s in stats_list])
        temporal_vars = np.array([s['output_temporal_var'] for s in stats_list])
        
        # Define thresholds (95th percentile)
        max_threshold = np.percentile(output_maxs, 95)
        range_threshold = np.percentile(output_ranges, 95)
        var_threshold = np.percentile(temporal_vars, 95)
        
        # Find anomalies
        anomaly_mask = (output_maxs > max_threshold) | \
                       (output_ranges > range_threshold) | \
                       (temporal_vars > var_threshold)
        
        anomaly_indices = [s['index'] for i, s in enumerate(stats_list) if anomaly_mask[i]]
        
        print(f"Total samples: {len(stats_list)}")
        print(f"Anomalous samples: {len(anomaly_indices)} ({len(anomaly_indices)/len(stats_list)*100:.2f}%)")
        print(f"\nThresholds (95th percentile):")
        print(f"  Max value: {max_threshold:.4f}")
        print(f"  Range: {range_threshold:.4f}")
        print(f"  Temporal variance: {var_threshold:.6f}")
        
        if len(anomaly_indices) > 0:
            print(f"\nFirst 20 anomalous indices: {anomaly_indices[:20]}")
            if len(anomaly_indices) > 20:
                print(f"  ... and {len(anomaly_indices)-20} more")
        
        return anomaly_indices
    
    def plot_distributions(self):
        """Plot distribution comparisons"""
        print(f"\n{'='*80}")
        print("Step 5: Generating Distribution Plots")
        print(f"{'='*80}")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training vs Testing Data Distribution Comparison', fontsize=16)
        
        metrics = [
            ('output_max', 'Max Value'),
            ('output_range', 'Value Range'),
            ('output_temporal_var', 'Temporal Variance'),
            ('output_mean', 'Mean Value'),
            ('output_std', 'Standard Deviation'),
        ]
        
        for idx, (key, label) in enumerate(metrics):
            if idx >= 6:
                break
            
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            train_vals = [s[key] for s in self.train_stats]
            test_vals = [s[key] for s in self.test_stats]
            
            ax.hist(train_vals, bins=50, alpha=0.5, label='Train', color='blue', density=True)
            ax.hist(test_vals, bins=50, alpha=0.5, label='Test', color='red', density=True)
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < 6:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('data_distribution_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: data_distribution_comparison.png")
        plt.close()
    
    def compare_extreme_samples(self):
        """Compare extreme samples between train and test"""
        print(f"\n{'='*80}")
        print("Step 6: Extreme Sample Analysis")
        print(f"{'='*80}")
        
        # Get top 5% extreme samples from each set
        train_maxs = [(s['index'], s['output_max']) for s in self.train_stats]
        test_maxs = [(s['index'], s['output_max']) for s in self.test_stats]
        
        train_maxs.sort(key=lambda x: x[1], reverse=True)
        test_maxs.sort(key=lambda x: x[1], reverse=True)
        
        n_extreme = max(int(len(train_maxs) * 0.05), 10)
        
        train_extreme = train_maxs[:n_extreme]
        test_extreme = test_maxs[:n_extreme]
        
        print(f"\nTop {n_extreme} extreme samples:")
        print(f"\nTraining Set:")
        print(f"  Max value range: {train_extreme[-1][1]:.4f} - {train_extreme[0][1]:.4f}")
        print(f"  Indices: {[idx for idx, _ in train_extreme[:10]]}")
        
        print(f"\nTesting Set:")
        print(f"  Max value range: {test_extreme[-1][1]:.4f} - {test_extreme[0][1]:.4f}")
        print(f"  Indices: {[idx for idx, _ in test_extreme[:10]]}")
        
        # Check if test has more extreme values than train
        train_max_val = train_extreme[0][1]
        test_max_val = test_extreme[0][1]
        
        if test_max_val > train_max_val * 1.1:
            print(f"\n⚠ WARNING: Test set has significantly higher extreme values!")
            print(f"  Train max: {train_max_val:.4f}")
            print(f"  Test max: {test_max_val:.4f}")
            print(f"  Difference: {(test_max_val/train_max_val - 1)*100:.1f}%")
            print(f"\n  This could explain poor performance on test extremes!")
        else:
            print(f"\n✓ Extreme values are comparable between train and test")
    
    def generate_report(self):
        """Generate final report"""
        print(f"\n{'='*80}")
        print("FINAL REPORT")
        print(f"{'='*80}")
        
        train_output_maxs = [s['output_max'] for s in self.train_stats]
        test_output_maxs = [s['output_max'] for s in self.test_stats]
        
        print(f"\nDataset Summary:")
        print(f"  Training samples: {len(self.train_stats)}")
        print(f"  Testing samples: {len(self.test_stats)}")
        
        print(f"\nOutput Value Statistics:")
        print(f"  Train - Mean: {np.mean(train_output_maxs):.4f}, Max: {np.max(train_output_maxs):.4f}")
        print(f"  Test  - Mean: {np.mean(test_output_maxs):.4f}, Max: {np.max(test_output_maxs):.4f}")
        
        # Key findings
        print(f"\n{'='*80}")
        print("KEY FINDINGS:")
        print(f"{'='*80}")
        
        train_anomaly_pct = sum(1 for s in self.train_stats if s['output_max'] > np.percentile(train_output_maxs, 95)) / len(self.train_stats) * 100
        test_anomaly_pct = sum(1 for s in self.test_stats if s['output_max'] > np.percentile(test_output_maxs, 95)) / len(self.test_stats) * 100
        
        print(f"\n1. Anomaly Distribution:")
        print(f"   Train: {train_anomaly_pct:.1f}% samples are extreme")
        print(f"   Test:  {test_anomaly_pct:.1f}% samples are extreme")
        
        if abs(train_anomaly_pct - test_anomaly_pct) > 2:
            print(f"   ⚠ Significant difference in anomaly rates!")
        else:
            print(f"   ✓ Similar anomaly rates")
        
        print(f"\n2. Data Quality:")
        train_nan = sum(s['output_nan_count'] for s in self.train_stats)
        test_nan = sum(s['output_nan_count'] for s in self.test_stats)
        print(f"   Train NaN count: {train_nan}")
        print(f"   Test NaN count: {test_nan}")
        
        if train_nan == 0 and test_nan == 0:
            print(f"   ✓ No NaN values detected")
        
        print(f"\n3. Conclusion:")
        if np.max(test_output_maxs) > np.max(train_output_maxs) * 1.1:
            print(f"   ⚠ Test set contains values outside training range")
            print(f"   → Model hasn't seen such extreme values during training")
            print(f"   → This explains poor performance on extreme test samples")
        else:
            print(f"   ✓ Test values are within training range")
            print(f"   → Poor performance on extremes is due to model capacity")
            print(f"   → Consider using robust loss functions (Huber Loss)")
        
        print(f"\n{'='*80}\n")

def main():
    print("="*80)
    print("DATA QUALITY CHECK")
    print("="*80)
    print("\nThis script will:")
    print("1. Load train and test indices")
    print("2. Scan all data files")
    print("3. Compare statistical distributions")
    print("4. Detect anomalies")
    print("5. Generate comparison plots")
    print("6. Provide diagnostic report")
    
    checker = DataQualityChecker()
    
    # Load indices
    checker.load_indices()
    
    # Scan datasets (use sample_rate=0.2 for quick check, 1.0 for full scan)
    print("\nNote: Using 20% sampling for quick check. Set sample_rate=1.0 for full scan.")
    checker.train_stats = checker.scan_dataset(checker.train_indices, is_train=True, sample_rate=0.2)
    checker.test_stats = checker.scan_dataset(checker.test_indices, is_train=False, sample_rate=1.0)
    
    # Analyze
    checker.analyze_statistics()
    checker.compare_extreme_samples()
    checker.plot_distributions()
    checker.generate_report()
    
    print("✓ Data quality check complete!")
    print("\nGenerated files:")
    print("  - data_distribution_comparison.png")

if __name__ == '__main__':
    main()
