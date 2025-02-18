import os

def update_paths_in_code(file_path):
    """
    Reads a Python file, updates occurrences of 'data' folder paths to 'Research_Data',
    and writes the changes back to the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    updated_lines = []
    for line in lines:
        # Update paths that reference the 'data' folder in path_parent
        updated_line = line.replace(
            "os.path.join(path_parent, 'data'", "os.path.join(path_parent, '..', 'Research_Data'"
        )
        updated_lines.append(updated_line)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(updated_lines)
    
    print(f"Updated paths in {file_path}")

# Get the current directory of this script
current_directory = os.path.dirname(os.path.abspath(__file__))

# List of files to update in the same directory
files_to_update = [
    "utils.py",
    "cnn_val_circle.py",
    "training_k_fold_2d.py",
    "svm_cross_validation.py",
    "cnn_validation.py",
    "channel_mapping_2d.py"
]

# Apply updates to each file
for file_name in files_to_update:
    file_path = os.path.join(current_directory, file_name)
    if os.path.exists(file_path):
        update_paths_in_code(file_path)
    else:
        print(f"File not found: {file_path}")