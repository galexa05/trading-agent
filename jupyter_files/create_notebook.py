#!/usr/bin/env python

import nbformat as nbf
from pathlib import Path

# Create a new notebook
nb = nbf.v4.new_notebook()

# Read the original Python file content
py_file = Path('summarization_agent_walkthrough_original.py')
content = py_file.read_text()

# Split the content into cells
cells = []
current_cell = []
cell_type = None

lines = content.split('\n')
i = 0

while i < len(lines):
    line = lines[i]
    
    # Check for cell markers
    if line.startswith('# Cell') and ' - ' in line:
        # Save the previous cell if it exists
        if current_cell and cell_type:
            if cell_type == 'markdown':
                cells.append(nbf.v4.new_markdown_cell('\n'.join(current_cell)))
            else:  # code cell
                cells.append(nbf.v4.new_code_cell('\n'.join(current_cell)))
        
        # Start a new cell
        current_cell = []
        cell_type = line.split(' - ')[1].lower()
        i += 1
        
        # Skip the triple quotes or code indicator
        if cell_type == 'markdown':
            if i < len(lines) and lines[i].startswith('"""'):
                i += 1  # Skip opening quotes
                
                # Get all content until closing quotes
                while i < len(lines) and not lines[i].endswith('"""'):
                    current_cell.append(lines[i])
                    i += 1
                
                if i < len(lines):  # Skip closing quotes
                    i += 1
        else:  # code cell
            # Gather all lines until the next cell marker or end of file
            while i < len(lines) and not (lines[i].startswith('# Cell') and ' - ' in lines[i]):
                current_cell.append(lines[i])
                i += 1
    else:
        i += 1

# Add the last cell
if current_cell and cell_type:
    if cell_type == 'markdown':
        cells.append(nbf.v4.new_markdown_cell('\n'.join(current_cell)))
    else:  # code cell
        cells.append(nbf.v4.new_code_cell('\n'.join(current_cell)))

# Add cells to the notebook
nb.cells = cells

# Write the notebook to a file
with open('summarization_agent_walkthrough_new.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully!")
