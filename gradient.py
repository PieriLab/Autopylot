import os
import time
import re
import shutil
from collections import deque
from argparse import ArgumentParser
from pathlib import Path
from molecule import Molecule  
from candidate import CandidateListGenerator
from launcher import launch_TCcalculation
from results_collector import exclude_TC_unsuccessful_calculations
import io_utils as io  

def read_single_arguments():
    """Parse input YAML file argument."""
    parser = ArgumentParser(description="Launch AutoPilot on a geometry.")
    parser.add_argument("-i", "--input_yaml", type=Path, required=True, help="Path of yaml input file")
    return parser.parse_args()

def run_gradient_calculations(yaml_file, timeout=4000000, interval=90):
    """Run gradient calculations for candidates in the YAML file and wait for all to complete."""
    settings = io.yload(yaml_file)
    n_singlets = settings['reference']['singlets']
    #n_triplets = settings['reference'].get('triplets', 0)
    fol_name = Path(yaml_file).absolute().parent
    charge = settings['general']['charge']
    geometry = settings['general']['coordinates']
    geom_file = fol_name / geometry
    mol = Molecule.from_xyz(geometry)
    nelec = mol.nelectrons - charge
    
    # Track only gradient calculation log files
    gradient_log_files = []

    if 'candidates' in settings:
        for calc_type in settings['candidates']:
            vee_settings = settings['candidates'][calc_type]
            vee_settings.update({
                'calc_type': calc_type,
                'nelec': nelec,
                'charge': charge,
                'n_singlets': n_singlets
            })

            case_settings = vee_settings | settings['candidates'][calc_type]
            candidates_list = CandidateListGenerator(**case_settings).create_candidate_list()
            completed_candidates = [
                candidate for candidate in candidates_list
                if exclude_TC_unsuccessful_calculations(fol_name, fol_name / f'{candidate.full_method}' / 'tc.out')
            ]

            for candidate in completed_candidates:
                gradient_folder_path = fol_name / f'gradient_{candidate.full_method}'
                os.makedirs(gradient_folder_path, exist_ok=True)

                #Filtering out excited state energy calcuation. Want only gradient. 
                gradient_calc_settings = candidate.calc_settings.copy()
                gradient_calc_settings['run'] = 'gradient'
                gradient_calc_settings['hhtdasinglets'] = 1
                gradient_calc_settings['cassinglets'] = 1

                #Retrieve c0 orbitals from appropriate files
                geom_name = geom_file.stem
                energy_calc_path = fol_name / candidate.full_method
                if 'casscf' in candidate.full_method.lower():
                    scr_location = energy_calc_path / f"scr.{geom_name}" / 'c0.casscf'
                elif 'casci' in candidate.full_method.lower():
                    scr_location = energy_calc_path / f"scr.{geom_name}" / 'c0'
                else:
                    scr_location = None

                if scr_location and scr_location.exists():
                    gradient_calc_settings['guess'] = str(scr_location)
                    if 'casscf' in candidate.full_method.lower():
                        gradient_calc_settings['casguess'] = str(scr_location)
                    print(f"Extracting orbitals from parent ({candidate.full_method}): {scr_location}")
                else:
                    gradient_calc_settings['guess'] = 'generate'
                    print(f"No orbitals found for {candidate.full_method}, optimizing orbitals, which may increase cost.")

                omitted = {'cisnumstates', 'cismax', 'cismaxiter', 'cisconvtol'}
                filtered_settings = {key: value for key, value in gradient_calc_settings.items() if key not in omitted}

                launch_TCcalculation(gradient_folder_path, geom_file, filtered_settings | settings['general'])
                
            # Add only valid gradient paths to gradient_log_files, with debug output
            print(f"Adding gradient job log: {gradient_folder_path / 'tc.out'}")
            gradient_log_files.append((gradient_folder_path / 'tc.out', gradient_folder_path))

    # Pass only gradient-specific log files to the wait_for_completion function
    failed_jobs, gradient_errors = wait_for_completion(gradient_log_files, timeout, interval)
    move_failed_jobs(failed_jobs, gradient_errors, fol_name)

def wait_for_completion(log_files, timeout, interval):
    """Wait for all gradient calculations to complete or timeout, returning lists of incomplete and error jobs."""
    start_time = time.time()
    error_pattern = re.compile(r'Job terminated', re.IGNORECASE)  # Only checking for "Job terminated"
    success_time_marker = "Total processing time"
    success_finish_marker = "Job finished:"
    failed_jobs = []
    gradient_errors = []

    def is_file_complete(file_path):
        """Check if the file has stopped growing to assume it's fully written."""
        
        while not file_path.exists():
            print(f"{file_path} not found yet. Waiting...")
            time.sleep(interval)

        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as file:
                found_success = False
                found_error = False

                # Check each line in the file for success or error markers
                for line in file:
                    if success_finish_marker in line:
                        found_success = True
                        break
                    if error_pattern.search(line):
                        found_error = True
                        break

                if found_success or found_error:
                    previous_size = os.path.getsize(file_path)
                    print(f"Existing log file found with marker. Initial file size: {previous_size}")
                else:
                    previous_size = -1  # File exists but does not contain markers
                    print("Existing log file found, but no success or error markers. Starting with size: -1")

        while time.time() - start_time < timeout:
            current_size = os.path.getsize(file_path)
            if current_size == previous_size:
                return True  # File size hasn't changed, assume complete
            previous_size = current_size
            time.sleep(interval)  # Wait and re-check size
        print(f"Timeout reached while waiting for file: {file_path}")
        return False   

    for log_file, gradient_folder_path in log_files:
        print(f"Checking log file: {log_file}, Folder: {gradient_folder_path}")

        # Ensure the directory exists before classification
        if not gradient_folder_path.exists():
            print(f"Directory not found for job: {gradient_folder_path}. Skipping classification.")
            continue

        try:
            # Wait until file is stable and likely complete
            if not is_file_complete(log_file):
                print(f"Waiting for file completion: {log_file}")
                continue

            # Read file contents after ensuring stability
            with open(log_file, 'r', encoding='utf-8') as file:
                contents = deque(file, maxlen=10)  # Keeps only the last 10 lines in memory
                print(f"Content of {log_file}:\n{contents}\n--- End of Content ---\n")  # Debug output

                # Check for success markers as substrings
                found_success_time = any(success_time_marker in line for line in contents)
                found_success_finish = any(success_finish_marker in line for line in contents)
                found_error = any(error_pattern.search(line) for line in contents)

                # Debugging output to track each condition's status
                print(f"found_success_time: {found_success_time}")
                print(f"found_success_finish: {found_success_finish}")
                print(f"found_error: {found_error}\n")

                # Prioritize success: if both success markers are found, consider it successful
                if found_success_time and found_success_finish:
                    print(f"Job completed successfully: {gradient_folder_path}")
                    continue  # Move to next job as this one is successful

                elif found_error:
                    gradient_errors.append((log_file, gradient_folder_path))
                    print(f"Job marked as error: {gradient_folder_path}")
                else:
                    # Mark as failed only if directory exists and success markers are missing
                    failed_jobs.append((log_file, gradient_folder_path))
                    print(f"Incomplete job added to failed_jobs: {gradient_folder_path}")

        except FileNotFoundError:
            print(f"Log file {log_file} not found, inspect cwd manually.")

    # Debug output to confirm final classification
    print(f"Final failed_jobs: {[path for _, path in failed_jobs]}")
    print(f"Final calculation_errors: {[path for _, path in gradient_errors]}")
    
    return failed_jobs, gradient_errors

def move_failed_jobs(failed_jobs, gradient_errors, fol_name):
    """Move failed gradient jobs to failed_gradient_jobs and their respective non-gradient jobs to SPEs_of_failed_gradients."""
    SPEs_of_failed_gradients = fol_name / "SPEs_of_failed_gradients"
    failed_gradient_jobs = fol_name / "failed_gradient_jobs"
    
    # Ensure directories are created
    SPEs_of_failed_gradients.mkdir(exist_ok=True)
    failed_gradient_jobs.mkdir(exist_ok=True)

    for log_file, gradient_folder_path in failed_jobs:
        # Check if the directory exists before attempting to move it
        if not gradient_folder_path.exists():
            print(f"Directory not found for failed job: {gradient_folder_path}. It may have been moved or deleted.")
            continue  # Skip moving if the directory doesn't exist

        if "gradient_" not in gradient_folder_path.name:
            print(f"Skipping non-gradient job in failed_jobs: {gradient_folder_path}")
            continue  # Skip moving non-gradient jobs to failed directories

        dest_gradient_path = failed_gradient_jobs / gradient_folder_path.name
        print(f"Moving failed gradient job {gradient_folder_path} to {dest_gradient_path}")
        shutil.move(str(gradient_folder_path), str(dest_gradient_path))

        # Move the associated non-gradient job to SPEs_of_failed_gradients
        non_gradient_directory = gradient_folder_path.parent / gradient_folder_path.name.replace("gradient_", "")
        if non_gradient_directory.exists():
            dest_spe_path = SPEs_of_failed_gradients / non_gradient_directory.name
            print(f"Moving associated non-gradient job {non_gradient_directory} to {dest_spe_path}")
            shutil.move(str(non_gradient_directory), str(dest_spe_path))
        else:
            print(f"Associated non-gradient directory not found: {non_gradient_directory}")

    for log_file, gradient_folder_path in gradient_errors:
        # Check if the directory exists before attempting to move it
        if not gradient_folder_path.exists():
            print(f"Directory not found for error job: {gradient_folder_path}. It may have been moved or deleted.")
            continue  # Skip moving if the directory doesn't exist

        if "gradient_" not in gradient_folder_path.name:
            print(f"Skipping non-gradient job in gradient_errors: {gradient_folder_path}")
            continue

        dest_gradient_path = failed_gradient_jobs / gradient_folder_path.name
        print(f"Moving gradient job with error {gradient_folder_path} to {dest_gradient_path}")
        shutil.move(str(gradient_folder_path), str(dest_gradient_path))

        non_gradient_directory = gradient_folder_path.parent / gradient_folder_path.name.replace("gradient_", "")
        if non_gradient_directory.exists():
            dest_spe_path = SPEs_of_failed_gradients / non_gradient_directory.name
            print(f"Moving associated non-gradient job {non_gradient_directory} to {dest_spe_path}")
            shutil.move(str(non_gradient_directory), str(dest_spe_path))
        else:
            print(f"Associated non-gradient directory not found: {non_gradient_directory}")

    print("Completed moving all failed and error-containing gradient jobs and their associated non-gradient jobs.")

def main():
    print(f"test")
    args = read_single_arguments()
    yaml_file = args.input_yaml
    run_gradient_calculations(yaml_file)

if __name__ == "__main__":
    main()
