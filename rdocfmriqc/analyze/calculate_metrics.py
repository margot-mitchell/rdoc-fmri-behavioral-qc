"""Calculates behavioral metrics from preprocessed parquet files.

This script reads preprocessed parquet files and calculates various metrics
for quality control and analysis.

Usage:
    python -m rdocfmriqc.analyze.calculate_metrics <subject_folder>
    Example: python -m rdocfmriqc.analyze.calculate_metrics sub-s01
"""

import argparse
import os
import sys
import logging
from typing import List

import polars as pl

from .tasks.longest_omission import append_longest_omission_metric
from .tasks import (
    ax_cpt_rdoc,
    cued_task_switching_rdoc,
    flanker_rdoc,
    go_nogo_rdoc,
    n_back_rdoc,
    operation_span_rdoc,
    simple_span_rdoc,
    spatial_cueing_rdoc,
    spatial_task_switching_rdoc,
    stop_signal_rdoc,
    stroop_rdoc,
    visual_search_rdoc,
    op_only_span_rdoc,
)

def get_function_mapping(task_dir):
    """Map task directory names to their corresponding analysis functions."""
    # Normalize task directory name to handle different variations
    normalized_dir = task_dir.lower().replace('-', '_')
    
    # Remove any suffixes like _practice, _rdoc, _fmri
    suffixes_to_remove = ['_practice', '_rdoc', '_fmri', '__fmri']
    for suffix in suffixes_to_remove:
        if normalized_dir.endswith(suffix):
            normalized_dir = normalized_dir[:-len(suffix)]
    
    # Handle special cases
    if normalized_dir in ['opspan', 'op_only_span']:
        return operation_span_rdoc
    elif normalized_dir in ['oponly', 'oponlyspan']:
        return op_only_span_rdoc
    elif normalized_dir in ['simplespan']:
        return simple_span_rdoc
    elif normalized_dir in ['stopsignal']:
        return stop_signal_rdoc
    elif normalized_dir in ['gonogo']:
        return go_nogo_rdoc
    elif normalized_dir in ['axcpt']:
        return ax_cpt_rdoc
    
    # Standard mappings
    mappings = {
        'ax_cpt': ax_cpt_rdoc,
        'flanker': flanker_rdoc,
        'go_nogo': go_nogo_rdoc,
        'nback': n_back_rdoc,
        'simple_span': simple_span_rdoc,
        'spatialcueing': spatial_cueing_rdoc,
        'cuedts': cued_task_switching_rdoc,
        'stop_signal': stop_signal_rdoc,
        'visualsearch': visual_search_rdoc,
        'stroop': stroop_rdoc,
        'operation_span': operation_span_rdoc,
        'spatialts': spatial_task_switching_rdoc,
    }
    return mappings.get(normalized_dir, None)

def main():
    # Configure logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate behavioral metrics')
    parser.add_argument('subject_folder', nargs='?', help='Subject folder to process (e.g., sub-s01)')
    parser.add_argument('--session', help='Specific session to process (e.g., ses-1, ses-pretouch)')
    args = parser.parse_args()

    if args.subject_folder:
        # Single subject mode
        subject_folders = [args.subject_folder]
    else:
        # All subjects mode
        preprocessed_dir = 'preprocessed_data'
        subject_folders = [f for f in os.listdir(preprocessed_dir)
                           if os.path.isdir(os.path.join(preprocessed_dir, f)) and f.startswith('sub-')]
        if not subject_folders:
            print(f"No subject folders found in {preprocessed_dir}")
            sys.exit(1)
        print(f"Processing all subjects: {', '.join(subject_folders)}")

    for subject_folder in subject_folders:
        print(f"\nProcessing subject folder: {subject_folder}")
        print(f"DEBUG: subject_folder variable = '{subject_folder}'")
        print(f"DEBUG: args.session = '{args.session}'")
        
        # Get all task directories for this subject
        preprocessed_dir = os.path.join('preprocessed_data', subject_folder)
        if not os.path.exists(preprocessed_dir):
            logging.error(f'Directory {preprocessed_dir} does not exist')
            continue

        # Get all task directories
        task_dirs = [d for d in os.listdir(preprocessed_dir)
                    if os.path.isdir(os.path.join(preprocessed_dir, d))]

        # Process each task
        for task_dir in task_dirs:
            print(f"  Processing task: {task_dir}")
            
            # Get the analysis function for this task
            analysis_func = get_function_mapping(task_dir)
            if analysis_func is None:
                logging.warning(f'Unknown task: {task_dir}')
                continue

            # Get all parquet files for this task
            task_path = os.path.join(preprocessed_dir, task_dir)
            parquet_files = [f for f in os.listdir(task_path) if f.endswith('.parquet')]

            if not parquet_files:
                logging.warning(f'No parquet files found in {task_path}')
                continue

            # Filter files by session if specified and process directly
            if args.session:
                session_files = [f for f in parquet_files if args.session in f]
                if not session_files:
                    logging.warning(f'No files found for session {args.session} in {task_path}')
                    continue
                print(f"    Processing {len(session_files)} files for session {args.session}")
                
                # Create output directory for this specific session
                output_dir = os.path.join('results', subject_folder, args.session, 'metrics')
                print(f"    DEBUG: Creating output directory: {output_dir}")
                print(f"    DEBUG: Components - results/{subject_folder}/{args.session}/metrics")
                os.makedirs(output_dir, exist_ok=True)

                task_metrics_list = []
                # Process each file for this session
                for parquet_file in session_files:
                    file_path = os.path.join(task_path, parquet_file)
                    try:
                        # Read the parquet file
                        df = pl.read_parquet(file_path)
                        
                        # Calculate metrics
                        metrics = analysis_func(df, filename=parquet_file)
                        metrics = append_longest_omission_metric(
                            metrics, df, task_dir, parquet_file
                        )

                        # Add task and file information
                        metrics = metrics.with_columns([
                            pl.lit(task_dir).alias('task'),
                            pl.lit(parquet_file).alias('file')
                        ])
                        
                        # Reorder columns to put metric first
                        metrics = metrics.select(['metric', 'value', 'task', 'file'])
                        
                        task_metrics_list.append(metrics)
                        
                    except Exception as e:
                        logging.error(f'Error processing {file_path}: {str(e)}')
                        continue

                # Save all metrics for this task to CSV
                if task_metrics_list:
                    task_metrics_df = pl.concat(task_metrics_list)
                    # Special case for operation span to ensure lowercase 'o'
                    if task_dir.lower() in ['opspan', 'op_only_span']:
                        metrics_csv_path = os.path.join(output_dir, 'opSpan_metrics.csv')
                    else:
                        metrics_csv_path = os.path.join(output_dir, f'{task_dir}_metrics.csv')
                    task_metrics_df.write_csv(metrics_csv_path)
                    print(f"    Saved metrics to {metrics_csv_path}")
            
            else:
                # No specific session - group files by session and process all
                session_files_map = {}
                for parquet_file in parquet_files:
                    # Extract session from filename (format: sub-sX_ses-Y_task-...)
                    session_name = None
                    parts = parquet_file.split('_')
                    for part in parts:
                        if part.startswith('ses-'):
                            session_name = part
                            break
                    
                    if session_name:
                        if session_name not in session_files_map:
                            session_files_map[session_name] = []
                        session_files_map[session_name].append(parquet_file)
                    else:
                        logging.warning(f'Could not extract session from filename: {parquet_file}')

                # Process each session's files
                for session_name, session_file_list in session_files_map.items():
                    print(f"    Processing {len(session_file_list)} files for session {session_name}")
                    
                    # Create output directory for this session
                    output_dir = os.path.join('results', subject_folder, session_name, 'metrics')
                    os.makedirs(output_dir, exist_ok=True)

                    task_metrics_list = []
                    # Process each file for this session
                    for parquet_file in session_file_list:
                        file_path = os.path.join(task_path, parquet_file)
                        try:
                            # Read the parquet file
                            df = pl.read_parquet(file_path)
                            
                            # Calculate metrics
                            metrics = analysis_func(df, filename=parquet_file)
                            metrics = append_longest_omission_metric(
                                metrics, df, task_dir, parquet_file
                            )

                            # Add task and file information
                            metrics = metrics.with_columns([
                                pl.lit(task_dir).alias('task'),
                                pl.lit(parquet_file).alias('file')
                            ])
                            
                            # Reorder columns to put metric first
                            metrics = metrics.select(['metric', 'value', 'task', 'file'])
                            
                            task_metrics_list.append(metrics)
                            
                        except Exception as e:
                            logging.error(f'Error processing {file_path}: {str(e)}')
                            continue

                    # Save all metrics for this task to CSV
                    if task_metrics_list:
                        task_metrics_df = pl.concat(task_metrics_list)
                        # Special case for operation span to ensure lowercase 'o'
                        if task_dir.lower() in ['opspan', 'op_only_span']:
                            metrics_csv_path = os.path.join(output_dir, 'opSpan_metrics.csv')
                        else:
                            metrics_csv_path = os.path.join(output_dir, f'{task_dir}_metrics.csv')
                        task_metrics_df.write_csv(metrics_csv_path)
                        print(f"    Saved metrics to {metrics_csv_path}")

if __name__ == '__main__':
    main()
