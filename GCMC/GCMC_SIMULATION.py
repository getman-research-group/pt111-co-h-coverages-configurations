import subprocess
import os
import shutil
import random
import math


# Constants
kB = 8.617333262145e-5
kB_S = 1.380649e-23  # Boltzmann constant in eV/K
T = 273 + 200  # Temperature in Kelvin
h = 6.62607015e-34
mu_O = -15.521775729470688  # Chemical potential for CO (eV)
mu_H = -3.824395409567156 # Chemical potential for H (eV)


# Define paths and initial structure
structure_base = "str_0"
temp_dir_base = "str_temp"

# Snapshot settings
SAVE_SNAPSHOTS = True       # Set to False to disable snapshot saving
SNAPSHOT_INTERVAL = 900000     # Save after every 100 accepted moves
SNAPSHOT_DIR_NAME = "SNAPS"

original_working_dir = os.getcwd()  # Save the original working directory

# Terminal command function
def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stderr.decode('utf-8'))
        return False
    return True

# Helper function to read energy file
def read_energy(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"Error: Energy file '{file_path}' does not exist.")
            raise FileNotFoundError(f"Energy file '{file_path}' not found.")

        with open(file_path, "r") as f:
            content = f.read().strip()  # Remove leading/trailing whitespaces
            
            print(f"Energy file content: '{content}'")  # Debugging: print the content

            if content:
                # Try converting to float
                energy_value = float(content)
                print(f"Successfully read energy value: {energy_value}")  # Debugging
                return energy_value
            else:
                raise ValueError(f"Energy file '{file_path}' is empty or contains invalid data.")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
        raise

    except ValueError as ve:
        print(f"Error reading energy file: {ve}")
        raise

# Function to initialize str_0
def initialize_structure(structure_dir):

    

    str_subdir = os.path.join(structure_dir)

    
    # Run initial corrdump and clusterpredict to populate energy file in str_0/str
    os.chdir(str_subdir)
    
    if not run_command("clusterpredict energy"):
        raise RuntimeError("Error running clusterpredict in str_0/str")

    os.chdir(original_working_dir)  # Go back to the original directory
    
    return

# Function to count particles (H and O) in str.out
def count_particles(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    count_H = content.count("H")
    count_O = content.count("O")
    return count_H, count_O

# Updated helper function to exchange particle positions (movement)
def exchange_particle(file_path, move_type):
    with open(file_path, "r") as f:
        lines = f.readlines()

    if move_type == "move_O":
        # Find positions of "0.610455 O" and "0.610455 Vac"
        positions_O = [i for i, line in enumerate(lines) if "0.610455 O" in line]
        positions_Vac = [i for i, line in enumerate(lines) if "0.610455 Vac" in line]
    elif move_type == "move_H":
        # Find positions of "0.599371 H" and "0.599371 Vac"
        positions_O = [i for i, line in enumerate(lines) if "0.599371 H" in line]
        positions_Vac = [i for i, line in enumerate(lines) if "0.599371 Vac" in line]

    if positions_O and positions_Vac:
        pos_O = random.choice(positions_O)
        pos_Vac = random.choice(positions_Vac)
        
        # Exchange positions of O or H with Vac
        if move_type == "move_O":
            lines[pos_O] = lines[pos_O].replace("0.610455 O", "0.610455 Vac")
            lines[pos_Vac] = lines[pos_Vac].replace("0.610455 Vac", "0.610455 O")
            print(f"Exchanged '0.610455 O' with '0.610455 Vac' between lines {pos_O + 1} and {pos_Vac + 1}")
        elif move_type == "move_H":
            lines[pos_O] = lines[pos_O].replace("0.599371 H", "0.599371 Vac")
            lines[pos_Vac] = lines[pos_Vac].replace("0.599371 Vac", "0.599371 H")
            print(f"Exchanged '0.599371 H' with '0.599371 Vac' between lines {pos_O + 1} and {pos_Vac + 1}")
        
        with open(file_path, "w") as f:
            f.writelines(lines)
        return True

    return False

# Save a copy of an accepted str_* folder at the requested interval
def save_snapshot(structure_dir, accepted_move_number):
    if not SAVE_SNAPSHOTS:
        return

    if accepted_move_number % SNAPSHOT_INTERVAL != 0:
        return

    snapshots_root = os.path.join(original_working_dir, SNAPSHOT_DIR_NAME)
    os.makedirs(snapshots_root, exist_ok=True)

    snapshot_name = f"str_{accepted_move_number}"
    snapshot_path = os.path.join(snapshots_root, snapshot_name)

    # Replace an existing snapshot with the same name, if present.
    if os.path.exists(snapshot_path):
        shutil.rmtree(snapshot_path)

    shutil.copytree(structure_dir, snapshot_path)
    print(f"Saved snapshot after {accepted_move_number} accepted moves: {snapshot_path}")


# Function to handle replacement and energy calculation loop with modified move options
def process_structure_loop(starting_structure, loop_index):
    temp_dir = os.path.join(original_working_dir, temp_dir_base)  # Use absolute path for temp_dir
    structure_dir = os.path.join(original_working_dir, starting_structure)  # Use absolute path for starting structure
    
    # Step 1: Copy structure folder as str_temp
    shutil.copytree(structure_dir, temp_dir)
    
    # Step 2: Enter str_temp/str and back up str.out before modifications
    str_subdir = os.path.join(temp_dir)
    str_out_path = os.path.join(str_subdir, "str.out")
    backup_str_out = f"{str_out_path}.backup"
    shutil.copy(str_out_path, backup_str_out)  # Create a backup of str.out
    
    accepted = False
    
    while not accepted:
        replaced = False
        
        # Step 3: Ensure a replacement is made before proceeding
        while not replaced:
            # Define move options explicitly, adding move_H
            move_options = [
                ("O_delete", "0.610455 O", "0.610455 Vac"),   # Deletion of O
                ("O_insert", "0.610455 Vac", "0.610455 O"),   # Insertion of O
                ("H_delete", "0.599371 H", "0.599371 Vac"),   # Deletion of H
                ("H_insert", "0.599371 Vac", "0.599371 H"),   # Insertion of H
                "move_O",                                     # Movement for O
                "move_H"                                      # Movement for H
            ]
            
            # Randomly choose a move from the list
            chosen_move = random.choice(move_options)
            
            # Handle particle movements separately
            if chosen_move == "move_O":
                replaced = exchange_particle(str_out_path, "move_O")  # Handle O movement
            elif chosen_move == "move_H":
                replaced = exchange_particle(str_out_path, "move_H")  # Handle H movement
            else:
                move_type, search_string, replace_string = chosen_move
                replaced = replace_instance(str_out_path, search_string, replace_string)

            if not replaced:
                print("No valid replacement or move made, trying again...")

        # Step 7: Count particles for N (H and O) in the current str.out
        N_H, N_O = count_particles(str_out_path)
        print(f"Number of H in current str.out = {N_H}, Number of O in current str.out = {N_O}")
        
        # Count particles for N (H and O) in the backup str.out
        backup_str_out_path = f"{str_out_path}.backup"
        N_H_OLD, N_O_OLD = count_particles(backup_str_out_path)
        print(f"Number of H in backup str.out = {N_H_OLD}, Number of O in backup str.out = {N_O_OLD}")

        # Step 4: Run corrdump and clusterpredict inside str_temp/str
        os.chdir(str_subdir)  # Change to str_temp/str
        if not run_command("clusterpredict energy"):
            raise RuntimeError("Error running clusterpredict in str_temp/str")
    
        
        # Step 6: Calculate energy difference after running clusterpredict in temp_dir/str
        energy_temp = read_energy(os.path.join(str_subdir, "energy"))  # Correct path to energy file

        # Calculate the delta with the energy of the previous structure
        if loop_index > 1:
            previous_energy_file = os.path.join(original_working_dir, f"str_{loop_index-1}", "energy")
        else:
            previous_energy_file = os.path.join(original_working_dir, "str_0", "energy")

        energy_prev = read_energy(previous_energy_file)
        
        # Use both current and backup particle counts for delta_E calculation
        delta_E = (energy_temp - energy_prev)
        print(f"Formation Energy difference (delta E) = {delta_E}")
        
        # Step 8: Apply Metropolis acceptance criterion based on move type
        if chosen_move == "move_O":  # O movement
            P_accept = metropolis_acceptance_move(delta_E)
            print(f"Actual P_accept for move O = {P_accept}")
        elif chosen_move == "move_H":  # H movement
            P_accept = metropolis_acceptance_move(delta_E)
            print(f"Actual P_accept for move H = {P_accept}")
        elif chosen_move[0] == "O_insert":  # Oxygen insertion
            P_accept = metropolis_acceptance_insert_O(delta_E, N_H_OLD, N_O_OLD)
            print(f"Actual P_accept for O_insert = {P_accept}")
        elif chosen_move[0] == "H_insert":  # Hydrogen insertion
            P_accept = metropolis_acceptance_insert_H(delta_E, N_H_OLD, N_O_OLD)
            print(f"Actual P_accept for H_insert = {P_accept}")
        elif chosen_move[0] == "O_delete":  # Oxygen deletion
            P_accept = metropolis_acceptance_delete_O(delta_E, N_H_OLD, N_O_OLD)
            print(f"Actual P_accept for O_delete = {P_accept}")
        elif chosen_move[0] == "H_delete":  # Hydrogen deletion
            P_accept = metropolis_acceptance_delete_H(delta_E, N_H_OLD, N_O_OLD)
            print(f"Actual P_accept for H_delete = {P_accept}")

        
        # Ensure P_accept does not exceed 1
        P_accept = min(P_accept, 1)

        random_num = random.uniform(0, 1)
        
        if P_accept >= random_num:
            print(f"Move accepted with P_accept = {P_accept} and delta E = {delta_E}")
            accepted = True
            os.remove("str.out.backup")
            os.chdir(original_working_dir)  # Move back to the original working directory
            # Rename str_temp to the next structure (e.g., str_1)
            new_dir = os.path.join(original_working_dir, f"str_{loop_index}")
            os.rename(temp_dir, new_dir)
            print(f"Renamed {temp_dir} to {new_dir}")

            # Save the accepted structure every SNAPSHOT_INTERVAL accepted moves.
            save_snapshot(new_dir, loop_index)

            # Delete older folders based on the current loop index
            if loop_index >= 50:
                folder_to_delete = os.path.join(original_working_dir, f"str_{loop_index - 50}")
                if os.path.exists(folder_to_delete):
                    shutil.rmtree(folder_to_delete)
                    print(f"Deleted folder: {folder_to_delete}")

            # Enter the new folder and read the energy file
            energy_file_path = os.path.join(new_dir, "energy")
            if os.path.exists(energy_file_path):
                with open(energy_file_path, "r") as energy_file:
                    energy_value = energy_file.read().strip()
                    with open("energy.out", "a") as energy_out:
                        energy_out.write(f"{energy_value}\n")
                    print(f"Recorded energy for structure {loop_index}: {energy_value}")
            else:
                print(f"Energy file not found in {new_dir}")

            # Read str.out and count occurrences of "O" and "H"
            str_out_path = os.path.join(new_dir, "str.out")
            if os.path.exists(str_out_path):
                with open(str_out_path, "r") as str_out_file:
                    content = str_out_file.read()
                    O_count = content.count("O")
                    H_count = content.count("H")
                    
                    with open("O.out", "a") as O_out:
                        O_out.write(f"{O_count}\n")
                    with open("H.out", "a") as H_out:
                        H_out.write(f"{H_count}\n")

                    print(f"Recorded O count ({O_count}) and H count ({H_count}) for structure {loop_index}")
            else:
                print(f"str.out file not found in {new_dir}")
        else:
            print(f"Move rejected with P_accept = {P_accept}, random value = {random_num}, and delta E = {delta_E}")
            restore_backup(str_out_path)  # Restore original str.out
        os.chdir(original_working_dir)  # Go back to the original directory



# Helper function to restore str.out after rejection
def restore_backup(file_path):
    backup_file = f"{file_path}.backup"
    if os.path.exists(backup_file):
        shutil.copy(backup_file, file_path)
    else:
        raise FileNotFoundError(f"Backup file '{backup_file}' not found.")

# Functions to handle metropolis acceptance for different moves
def metropolis_acceptance_insert_O(delta_E, N_H_OLD, N_O_OLD):
    return min(1, math.exp(-(delta_E - mu_O) / (kB * T)))

def metropolis_acceptance_insert_H(delta_E, N_H_OLD, N_O_OLD):
    return min(1, math.exp(-(delta_E - mu_H) / (kB * T)))

def metropolis_acceptance_delete_O(delta_E, N_H_OLD, N_O_OLD):
    return min(1, math.exp(-(delta_E + mu_O) / (kB * T)))

def metropolis_acceptance_delete_H(delta_E, N_H_OLD, N_O_OLD):
    return min(1, math.exp(-(delta_E + mu_H) / (kB * T)))

def metropolis_acceptance_move(delta_E):
    return min(1, math.exp(-delta_E / (kB * T)))




# Function to replace random instances and echo what was replaced
def replace_instance(file_path, search_string, replace_string):
    with open(file_path, "r") as f:
        lines = f.readlines()

    occurrences = [i for i, line in enumerate(lines) if search_string in line]
    if occurrences:
        random_occurrence = random.choice(occurrences)
        lines[random_occurrence] = lines[random_occurrence].replace(search_string, replace_string)
        print(f"Replaced '{search_string}' with '{replace_string}' at line {random_occurrence + 1}")
        with open(file_path, "w") as f:
            f.writelines(lines)
        return True
    return False

# Main loop for the entire process
structure_file = "str.out"
initial_structure = "str_0"

# First process for str_0/str
initialize_structure(initial_structure)

# Read the initial energy from str/energy (inside str_0/str)
try:
    energy_0 = read_energy(os.path.join(initial_structure, "energy"))  # Corrected path to energy file
except Exception as e:
    print(f"Failed to read energy: {e}")
    exit(1)

# Loop for subsequent iterations (e.g., from str_1 to str_n)
for i in range(1, 10001):
    print(f"Processing loop {i}...")
    process_structure_loop(f"str_{i-1}" if i > 1 else "str_0", i)

# Explicitly exit after completing all loops
print("All loops completed. Exiting the script.")
import sys
sys.exit(0)
