import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_data(file_path):
    """Load the performance data from CSV file."""
    try:
        # Add column names since the CSV doesn't have headers
        column_names = ['vertices', 'processes', 'threads', 'edge_density', 'change_ratio', 'time_ms', 'affected_vertices']
        data = pd.read_csv(file_path, names=column_names)
        print(f"Loaded data with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def analyze_process_scaling(data):
    """Analyze and plot the effect of increasing MPI processes."""
    # Filter data for process scaling (same vertices, edge density, change ratio)
    df = data[
        (data['vertices'] == 100) & 
        (data['threads'] == 2) & 
        (data['edge_density'] == 0.1) & 
        (data['change_ratio'] == 0.01)
    ].drop_duplicates(subset=['processes'])
    
    if len(df) < 2:
        print("Not enough data for process scaling analysis.")
        return
    
    df = df.sort_values('processes')
    
    # Calculate speedup relative to single process
    baseline = df[df['processes'] == 1]['time_ms'].values[0]
    df['speedup'] = baseline / df['time_ms']
    df['efficiency'] = df['speedup'] / df['processes']
    
    # Print analysis
    print("\n=== MPI Process Scaling Analysis ===")
    print(df[['processes', 'time_ms', 'speedup', 'efficiency']].to_string(index=False))
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    plt.plot(df['processes'], df['speedup'], 'o-', color='blue', linewidth=2, markersize=8)
    plt.axline((1, 1), slope=1, color='gray', linestyle='--', label='Ideal Speedup')
    plt.title('Speedup vs Number of MPI Processes')
    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.legend()
    plt.savefig('process_scaling.png')
    plt.close()

def analyze_thread_scaling(data):
    """Analyze and plot the effect of increasing OpenMP threads."""
    # Filter data for thread scaling (same vertices, processes, edge density, change ratio)
    df = data[
        (data['vertices'] == 100) & 
        (data['processes'] == 1) & 
        (data['edge_density'] == 0.1) & 
        (data['change_ratio'] == 0.01)
    ].drop_duplicates(subset=['threads'])
    
    if len(df) < 2:
        print("Not enough data for thread scaling analysis.")
        return
    
    df = df.sort_values('threads')
    
    # Calculate speedup relative to single thread
    baseline = df[df['threads'] == 1]['time_ms'].values[0]
    df['speedup'] = baseline / df['time_ms']
    df['efficiency'] = df['speedup'] / df['threads']
    
    # Print analysis
    print("\n=== OpenMP Thread Scaling Analysis ===")
    print(df[['threads', 'time_ms', 'speedup', 'efficiency']].to_string(index=False))
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    plt.plot(df['threads'], df['speedup'], 'o-', color='green', linewidth=2, markersize=8)
    plt.axline((1, 1), slope=1, color='gray', linestyle='--', label='Ideal Speedup')
    plt.title('Speedup vs Number of OpenMP Threads')
    plt.xlabel('Number of OpenMP Threads')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.legend()
    plt.savefig('thread_scaling.png')
    plt.close()

def analyze_size_scaling(data):
    """Analyze and plot the effect of increasing graph size."""
    # Filter data for size scaling (same processes, threads, edge density, change ratio)
    df = data[
        (data['processes'] == 2) & 
        (data['threads'] == 2) & 
        (data['edge_density'] == 0.1) & 
        (data['change_ratio'] == 0.01)
    ].drop_duplicates(subset=['vertices'])
    
    if len(df) < 2:
        print("Not enough data for size scaling analysis.")
        return
    
    df = df.sort_values('vertices')
    
    # Print analysis
    print("\n=== Problem Size Scaling Analysis ===")
    print(df[['vertices', 'time_ms']].to_string(index=False))
    
    # Plot execution time vs graph size
    plt.figure(figsize=(10, 6))
    plt.plot(df['vertices'], df['time_ms'], 'o-', color='red', linewidth=2, markersize=8)
    plt.title('Execution Time vs Graph Size')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True)
    plt.savefig('size_scaling.png')
    plt.close()

def analyze_edge_density(data):
    """Analyze and plot the effect of increasing edge density."""
    # Filter data for edge density analysis
    df = data[
        (data['vertices'] == 100) & 
        (data['processes'] == 2) & 
        (data['threads'] == 2) & 
        (data['change_ratio'] == 0.01)
    ].drop_duplicates(subset=['edge_density'])
    
    if len(df) < 2:
        print("Not enough data for edge density analysis.")
        return
    
    df = df.sort_values('edge_density')
    
    # Print analysis
    print("\n=== Edge Density Analysis ===")
    print(df[['edge_density', 'time_ms']].to_string(index=False))
    
    # Plot execution time vs edge density
    plt.figure(figsize=(10, 6))
    plt.plot(df['edge_density'], df['time_ms'], 'o-', color='purple', linewidth=2, markersize=8)
    plt.title('Execution Time vs Edge Density')
    plt.xlabel('Edge Density')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True)
    plt.savefig('edge_density.png')
    plt.close()

def analyze_change_ratio(data):
    """Analyze and plot the effect of increasing change ratio."""
    # Filter data for change ratio analysis
    df = data[
        (data['vertices'] == 100) & 
        (data['processes'] == 2) & 
        (data['threads'] == 2) & 
        (data['edge_density'] == 0.1)
    ].drop_duplicates(subset=['change_ratio'])
    
    if len(df) < 2:
        print("Not enough data for change ratio analysis.")
        return
    
    df = df.sort_values('change_ratio')
    
    # Print analysis
    print("\n=== Change Ratio Analysis ===")
    print(df[['change_ratio', 'time_ms']].to_string(index=False))
    
    # Plot execution time vs change ratio
    plt.figure(figsize=(10, 6))
    plt.plot(df['change_ratio'], df['time_ms'], 'o-', color='orange', linewidth=2, markersize=8)
    plt.title('Execution Time vs Change Ratio')
    plt.xlabel('Change Ratio')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True)
    plt.savefig('change_ratio.png')
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <performance_results.csv>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    data = load_data(data_file)
    
    # Run analyses
    analyze_process_scaling(data)
    analyze_thread_scaling(data)
    analyze_size_scaling(data)
    analyze_edge_density(data)
    analyze_change_ratio(data)
    
    print("\nPerformance analysis completed. Results saved as PNG files.")

if __name__ == "__main__":
    main() 