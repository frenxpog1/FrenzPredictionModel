import os
import sys
import json
import ast
import nbformat
from typing import List, Tuple

def check_python_syntax(file_path: str) -> List[Tuple[int, str]]:
    errors = []
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
    except SyntaxError as e:
        errors.append((e.lineno, e.msg))
    except Exception as e:
        errors.append((0, str(e)))
    return errors

def check_notebook_syntax(file_path: str) -> List[Tuple[int, str, str]]:
    errors = []
    try:
        with open(file_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                try:
                    ast.parse(cell.source)
                except SyntaxError as e:
                    # Adjust line number to be relative to the cell
                    errors.append((i, e.lineno, e.msg))
    except json.JSONDecodeError as e:
        errors.append((-1, e.lineno, f"Corrupted JSON: {e.msg}"))
    except Exception as e:
        errors.append((-2, 0, str(e)))
    return errors

def main():
    print("🛡️ Sentinel V1: Syntax & Integrity Guardian")
    print("="*40)
    
    targets = [
        '1_NoteBook/Prediction.ipynb',
        'features.py',
        'models.py',
        'model_zoo.py',
        'scraper.py'
    ]
    
    failed = False
    for target in targets:
        if not os.path.exists(target):
            continue
            
        print(f"Checking {target}...", end=" ")
        if target.endswith('.ipynb'):
            nb_errors = check_notebook_syntax(target)
            if nb_errors:
                print("❌ FAILED")
                for cell_idx, line, msg in nb_errors:
                    if cell_idx == -1:
                        print(f"  [CRITICAL] JSON Error at line {line}: {msg}")
                    else:
                        print(f"  [CELL {cell_idx}] Syntax Error at line {line}: {msg}")
                failed = True
            else:
                print("✅ OK")
        else:
            py_errors = check_python_syntax(target)
            if py_errors:
                print("❌ FAILED")
                for line, msg in py_errors:
                    print(f"  Line {line}: {msg}")
                failed = True
            else:
                print("✅ OK")
                
    if failed:
        sys.exit(1)
    else:
        print("\n✨ All systems clear.")

if __name__ == "__main__":
    main()
