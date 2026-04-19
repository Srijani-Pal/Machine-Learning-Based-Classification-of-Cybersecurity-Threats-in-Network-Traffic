"""
Script to merge CSV files from each folder using direct file concatenation
Ultra-efficient approach for very large datasets
"""
import os

# Define dataset directories
data_dirs = {
    'CIC-IDS-2017': 'd:/Project related/Datasets/CIC-IDS- 2017',
    'DDOS': 'd:/Project related/Datasets/DDOS',
    'NSL_KDD': 'd:/Project related/Datasets/NSL_KDD',
    'UNSW': 'd:/Project related/Datasets/UNSW'
}

output_dir = 'd:/Project related/Datasets/merged_datasets'
os.makedirs(output_dir, exist_ok=True)

print("Starting dataset merging (ultra-fast file concatenation)...\n")

def merge_csv_files(input_dir, output_file):
    """Merge CSV files by direct file concatenation - fastest method"""
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    
    if not csv_files:
        return 0
    
    total_rows = 0
    first_file = True
    
    with open(output_file, 'w', buffering=1024*1024) as outf:  # 1MB buffer
        for csv_file in csv_files:
            filepath = os.path.join(input_dir, csv_file)
            print(f"   Processing {csv_file}...")
            
            with open(filepath, 'r') as inf:
                # Copy header only from first file
                header = inf.readline()
                if first_file:
                    outf.write(header)
                    total_rows += 1
                    first_file = False
                
                # Copy all data lines
                for line in inf:
                    outf.write(line)
                    total_rows += 1
    
    return total_rows

# 1. Merge CIC-IDS-2017 files
print("1. Merging CIC-IDS-2017 files...")
try:
    cic_dir = data_dirs['CIC-IDS-2017']
    cic_output = os.path.join(output_dir, 'cic_ids_merged.csv')
    rows = merge_csv_files(cic_dir, cic_output)
    print(f"   ✓ CIC-IDS merged: {rows} rows")
except Exception as e:
    print(f"   ✗ Error merging CIC-IDS: {e}")

# 2. Merge DDOS files
print("\n2. Merging DDOS files...")
try:
    ddos_dir = data_dirs['DDOS']
    ddos_output = os.path.join(output_dir, 'ddos_merged.csv')
    rows = merge_csv_files(ddos_dir, ddos_output)
    print(f"   ✓ DDOS merged: {rows} rows")
except Exception as e:
    print(f"   ✗ Error merging DDOS: {e}")

# 3. Merge NSL_KDD files
print("\n3. Merging NSL_KDD files...")
try:
    nsl_dir = data_dirs['NSL_KDD']
    nsl_output = os.path.join(output_dir, 'nsl_kdd_merged.csv')
    rows = merge_csv_files(nsl_dir, nsl_output)
    print(f"   ✓ NSL_KDD merged: {rows} rows")
except Exception as e:
    print(f"   ✗ Error merging NSL_KDD: {e}")

# 4. Merge UNSW files
print("\n4. Merging UNSW files...")
try:
    unsw_dir = data_dirs['UNSW']
    unsw_output = os.path.join(output_dir, 'unsw_merged.csv')
    rows = merge_csv_files(unsw_dir, unsw_output)
    print(f"   ✓ UNSW merged: {rows} rows")
except Exception as e:
    print(f"   ✗ Error merging UNSW: {e}")

print(f"\n✓ All datasets merged successfully in {output_dir}")
