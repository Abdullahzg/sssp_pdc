import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_data(file_path):
    """Load the performance data from CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def plot_process_scaling(data):
    """Plot speedup vs number of processes."""
    # Filter data for process scaling tests
    process_data = data[
        (data['vertices'] == 500) & 
        (data['threads'] == 2) & 
        (data['edge_density'] == 0.1) & 
        (data['change_ratio'] == 0.01)
    ]
    
    if len(process_data) == 0:
        print("No data available for process scaling plot.")
        return
    
    # Calculate speedup relative to single process
    baseline = process_data[process_data['processes'] == 1]['time_ms'].values[0]
    process_data['speedup'] = baseline / process_data['time_ms']
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(process_data['processes'], process_data['speedup'], 'o-', linewidth=2, markersize=8)
    plt.axline((1, 1), slope=1, color='gray', linestyle='--', label='Ideal Speedup')
    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of MPI Processes')
    plt.grid(True)
    plt.legend()
    plt.savefig('process_scaling.png')
    plt.close()
    
    print("Process scaling plot saved as process_scaling.png")

def plot_thread_scaling(data):
    """Plot speedup vs number of OpenMP threads."""
    # Filter data for thread scaling tests
    thread_data = data[
        (data['vertices'] == 500) & 
        (data['processes'] == 2) & 
        (data['edge_density'] == 0.1) & 
        (data['change_ratio'] == 0.01)
    ]
    
    if len(thread_data) == 0:
        print("No data available for thread scaling plot.")
        return
    
    # Calculate speedup relative to single thread
    baseline = thread_data[thread_data['threads'] == 1]['time_ms'].values[0]
    thread_data['speedup'] = baseline / thread_data['time_ms']
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(thread_data['threads'], thread_data['speedup'], 'o-', linewidth=2, markersize=8)
    plt.axline((1, 1), slope=1, color='gray', linestyle='--', label='Ideal Speedup')
    plt.xlabel('Number of OpenMP Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of OpenMP Threads')
    plt.grid(True)
    plt.legend()
    plt.savefig('thread_scaling.png')
    plt.close()
    
    print("Thread scaling plot saved as thread_scaling.png")

def plot_size_scaling(data):
    """Plot execution time vs graph size."""
    # Filter data for size scaling tests
    size_data = data[
        (data['processes'] == 2) & 
        (data['threads'] == 4) & 
        (data['edge_density'] == 0.1) & 
        (data['change_ratio'] == 0.01)
    ]
    
    if len(size_data) == 0:
        print("No data available for size scaling plot.")
        return
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(size_data['vertices'], size_data['time_ms'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Vertices')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time vs Graph Size')
    plt.grid(True)
    plt.savefig('size_scaling.png')
    plt.close()
    
    print("Size scaling plot saved as size_scaling.png")

def plot_density_impact(data):
    """Plot execution time vs edge density."""
    # Filter data for density tests
    density_data = data[
        (data['vertices'] == 500) & 
        (data['processes'] == 2) & 
        (data['threads'] == 4) & 
        (data['change_ratio'] == 0.01)
    ]
    
    if len(density_data) == 0:
        print("No data available for density impact plot.")
        return
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(density_data['edge_density'], density_data['time_ms'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Edge Density')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time vs Edge Density')
    plt.grid(True)
    plt.savefig('density_impact.png')
    plt.close()
    
    print("Density impact plot saved as density_impact.png")

def plot_change_ratio_impact(data):
    """Plot execution time vs change ratio."""
    # Filter data for change ratio tests
    change_data = data[
        (data['vertices'] == 500) & 
        (data['processes'] == 2) & 
        (data['threads'] == 4) & 
        (data['edge_density'] == 0.1)
    ]
    
    if len(change_data) == 0:
        print("No data available for change ratio impact plot.")
        return
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(change_data['change_ratio'], change_data['time_ms'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Change Ratio')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time vs Change Ratio')
    plt.grid(True)
    plt.savefig('change_ratio_impact.png')
    plt.close()
    
    print("Change ratio impact plot saved as change_ratio_impact.png")

def generate_summary_table(data):
    """Generate a summary table of the performance results."""
    # Calculate speedups for process and thread scaling
    process_data = data[
        (data['vertices'] == 500) & 
        (data['threads'] == 2) & 
        (data['edge_density'] == 0.1) & 
        (data['change_ratio'] == 0.01)
    ]
    
    thread_data = data[
        (data['vertices'] == 500) & 
        (data['processes'] == 2) & 
        (data['edge_density'] == 0.1) & 
        (data['change_ratio'] == 0.01)
    ]
    
    # Create summary table
    summary = pd.DataFrame()
    
    if len(process_data) > 0:
        baseline_process = process_data[process_data['processes'] == 1]['time_ms'].values[0]
        process_data['speedup'] = baseline_process / process_data['time_ms']
        
        summary = pd.concat([summary, pd.DataFrame({
            'Test Type': 'Process Scaling',
            'Configuration': [f"{p} processes, 2 threads" for p in process_data['processes']],
            'Time (ms)': process_data['time_ms'],
            'Speedup': process_data['speedup'],
            'Efficiency': process_data['speedup'] / process_data['processes']
        })])
    
    if len(thread_data) > 0:
        baseline_thread = thread_data[thread_data['threads'] == 1]['time_ms'].values[0]
        thread_data['speedup'] = baseline_thread / thread_data['time_ms']
        
        summary = pd.concat([summary, pd.DataFrame({
            'Test Type': 'Thread Scaling',
            'Configuration': [f"2 processes, {t} threads" for t in thread_data['threads']],
            'Time (ms)': thread_data['time_ms'],
            'Speedup': thread_data['speedup'],
            'Efficiency': thread_data['speedup'] / thread_data['threads']
        })])
    
    if len(summary) > 0:
        summary.to_csv('performance_summary.csv', index=False)
        print("Summary table saved as performance_summary.csv")
        print("\nPerformance Summary:")
        print(summary.to_string(index=False))

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_performance.py <performance_results.csv>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    data = load_data(data_file)
    
    # Generate plots
    plot_process_scaling(data)
    plot_thread_scaling(data)
    plot_size_scaling(data)
    plot_density_impact(data)
    plot_change_ratio_impact(data)
    
    # Generate summary table
    generate_summary_table(data)

if __name__ == "__main__":
    main() 