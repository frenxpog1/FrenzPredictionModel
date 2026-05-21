import json
import os

def generate_notebook():
    notebook_path = "1_NoteBook/Prediction.ipynb"
    copy_code_path = "copy_code.py"
    
    if not os.path.exists(copy_code_path):
        print(f"❌ Error: {copy_code_path} not found.")
        return
        
    with open(copy_code_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Process lines to make paths dynamic using csv_dir
    modified_lines = []
    
    # Insert path resolution logic at the very top
    modified_lines.append("import os\n")
    modified_lines.append("csv_dir = 'csv_data' if os.path.exists('csv_data') else '../csv_data'\n")
    modified_lines.append("print(f'Using resolved CSV directory: {csv_dir}')\n\n")
    
    for line in lines:
        # Replace matches_df = pd.read_csv('csv_data/matches.csv')
        if "matches_df = pd.read_csv('csv_data/matches.csv')" in line:
            line = line.replace("'csv_data/matches.csv'", "f'{csv_dir}/matches.csv'")
        # Replace games_df   = pd.read_csv('csv_data/games.csv')
        elif "games_df   = pd.read_csv('csv_data/games.csv')" in line:
            line = line.replace("'csv_data/games.csv'", "f'{csv_dir}/games.csv'")
        # Replace rosters_df = pd.read_csv('csv_data/season_rosters.csv')
        elif "rosters_df = pd.read_csv('csv_data/season_rosters.csv')" in line:
            line = line.replace("'csv_data/season_rosters.csv'", "f'{csv_dir}/season_rosters.csv'")
        # Replace patches_df = pd.read_csv('csv_data/patches.csv')
        elif "patches_df = pd.read_csv('csv_data/patches.csv')" in line:
            line = line.replace("'csv_data/patches.csv'", "f'{csv_dir}/patches.csv'")
        # Replace final_matrix.to_csv('csv_data/ML_Feature_Matrix.csv', index=False)
        elif "final_matrix.to_csv('csv_data/ML_Feature_Matrix.csv', index=False)" in line:
            line = line.replace("'csv_data/ML_Feature_Matrix.csv'", "f'{csv_dir}/ML_Feature_Matrix.csv'")
        # Replace df = pd.read_csv('csv_data/ML_Feature_Matrix.csv')
        elif "df = pd.read_csv('csv_data/ML_Feature_Matrix.csv')" in line:
            line = line.replace("'csv_data/ML_Feature_Matrix.csv'", "f'{csv_dir}/ML_Feature_Matrix.csv'")
        # Replace 'csv_data/prediction_tracker.csv', index=False
        elif "'csv_data/prediction_tracker.csv', index=False" in line:
            line = line.replace("'csv_data/prediction_tracker.csv'", "f'{csv_dir}/prediction_tracker.csv'")
        # Replace print(f"\n💾 Full tracker saved to csv_data/prediction_tracker.csv")
        elif "saved to csv_data/prediction_tracker.csv" in line:
            line = line.replace("csv_data/prediction_tracker.csv", "{csv_dir}/prediction_tracker.csv")
            
        modified_lines.append(line)
        
    # Load existing notebook to preserve structure (cell 0 etc)
    if os.path.exists(notebook_path):
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    else:
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "install_cell",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "%pip install pandas scikit-learn xgboost\n",
                        "%pip install lightgbm\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "main_cell",
                    "metadata": {},
                    "outputs": [],
                    "source": []
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5
        }
        
    # Put the entire modified script into Cell 1 (index 1)
    # Ensure all strings end with newline
    nb["cells"][1]["source"] = [l if l.endswith('\n') else l + '\n' for l in modified_lines]
    
    # Save the updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"🎉 Successfully regenerated {notebook_path} with {len(modified_lines)} lines of clean code!")

if __name__ == "__main__":
    generate_notebook()
