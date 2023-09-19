# Define the keywords to filter
keywords = ["ground truth values","final","observation"]

# Open the file in read mode
file_path = "firstout.txt"  # Replace with the actual file path
filtered_lines = []

with open(file_path, "r") as file:
    for line in file:
        if any(line.startswith(keyword) for keyword in keywords):
            filtered_lines.append(line)

file_path = "outmeasure.txt"  # Replace with the actual file path
# Open the file in write mode and overwrite its content
with open(file_path, "w") as file:
    file.writelines(filtered_lines)
