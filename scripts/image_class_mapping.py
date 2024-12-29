import os
import csv
import sys


def generate_file_class_mapping(root_dir, output_file):
    """
    Recursively traverses the root directory and generates a CSV file.

    The CSV file contains:
    - First column: Relative path of each file.
    - Second column: Parent directory of each file.

    Args:
        root_dir (str): The root directory to traverse.
        output_file (str): The output CSV file path.
    """
    try:
        class_mapping_path_file_path = "LOC_synset_mapping.txt"

        # Initialize an empty dictionary to store the mapping
        class_mapping = {}

        # Open the file and parse it
        with open(class_mapping_path_file_path, "r") as file:
            for line_number, line in enumerate(file):
                # Split the line to get the first value
                first_value = line.split()[0].strip()
                # Map the first value to the line number
                class_mapping[first_value] = line_number

        # Open the output file for writing
        with open(output_file, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header row

            # Recursively traverse the directory
            for dirpath, _, filenames in os.walk(root_dir):
                for file in filenames:
                    # Get the relative path for the file
                    relative_path = os.path.relpath(os.path.join(dirpath, file), root_dir)
                    # Get the parent directory of the file
                    parent_dir = os.path.basename(dirpath)
                    class_idx = class_mapping[parent_dir]
                    # Write the row to the CSV
                    writer.writerow([relative_path, class_idx])

        print(f"Report generated successfully: {output_file}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python generate_file_report.py <root_directory> <output_csv_file>")
        sys.exit(1)

    # Parse command-line arguments
    root_directory = sys.argv[1]
    output_csv_file = sys.argv[2]

    # Generate the report
    generate_file_class_mapping(root_directory, output_csv_file)