import re
import os

# Read the input text file
with open('output.txt', 'r') as f:
    content = f.read()

# Define the delimiter pattern
delimiter = '#' * 180

# Split the content using the delimiter as a regular expression
sections = re.split(rf'(?={re.escape(delimiter)})', content)

# Remove any empty sections
sections = [section.strip() for section in sections if section.strip()]

print(len(sections))

# Create a directory to store the output files
output_directory = 'output_sections'
os.makedirs(output_directory, exist_ok=True)

# Write each section to separate files
for index, section in enumerate(sections, start=1):
    output_file_path = os.path.join(output_directory, f'output_{index}.txt')
    with open(output_file_path, 'w') as f:
        f.write(section)

print(f"{len(sections)} sections have been split and saved in the '{output_directory}' directory.")
