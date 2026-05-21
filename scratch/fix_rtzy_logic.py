import nbformat
import os

def fix_rtzy():
    nb_path = '1_NoteBook/Prediction.ipynb'
    with open(nb_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # RTzy is a Midlaner (RSG), KarlTzy is a Jungler (Liquid/Bren). 
    # Must keep them separate!
    
    setup_cell = nb.cells[0]
    source = setup_cell.source
    
    # Remove the bad mapping
    new_source = source.replace("'rtzy': 'karl tzy',", "")
    new_source = new_source.replace("'karl tzy': 'karl tzy',", "")
    
    setup_cell.source = new_source
    
    with open(nb_path, 'w') as f:
        nbformat.write(nb, f)

fix_rtzy()
print('Fixed: RTzy and KarlTzy are now separate players in the notebook.')
