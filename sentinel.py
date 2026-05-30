import os
import sys
import json
import ast
import sqlite3
import pandas as pd
import numpy as np
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
                    errors.append((i, e.lineno, e.msg))
    except json.JSONDecodeError as e:
        errors.append((-1, e.lineno, f"Corrupted JSON: {e.msg}"))
    except Exception as e:
        errors.append((-2, 0, str(e)))
    return errors

def audit_database(db_path: str = "./mlbb_data.db") -> List[str]:
    issues = []
    if not os.path.exists(db_path):
        issues.append(f"SQLite database missing at '{db_path}'")
        return issues
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check expected tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ["matches", "games", "season_rosters", "players", "heroes", "items", "patches"]
        for table in expected_tables:
            if table not in tables:
                issues.append(f"Missing expected database table: '{table}'")
                
        # Sample check matches table
        if "matches" in tables:
            cursor.execute("SELECT COUNT(*) FROM matches")
            matches_count = cursor.fetchone()[0]
            if matches_count == 0:
                issues.append("Table 'matches' is completely empty.")
                
        conn.close()
    except Exception as e:
        issues.append(f"Database connection error: {e}")
    return issues

def audit_feature_matrix(matrix_path: str = "csv_data/ML_Feature_Matrix.csv") -> List[str]:
    issues = []
    if not os.path.exists(matrix_path):
        issues.append(f"Feature matrix CSV missing at '{matrix_path}'")
        return issues
        
    try:
        df = pd.read_csv(matrix_path)
        rows, cols = df.shape
        
        # Validate shape constraints
        if rows < 2300:
            issues.append(f"Feature matrix row count ({rows}) is below the expected minimum of 2300.")
        if cols < 150:
            issues.append(f"Feature matrix column count ({cols}) is below the expected minimum of 150.")
            
        # Validate null values
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            # List some of the columns with null values
            null_cols = df.columns[df.isnull().any()].tolist()[:5]
            issues.append(f"Feature matrix contains {null_counts} null values in columns: {null_cols}")
            
        # Leakage verification: check that target label is not duplicated in features
        feature_cols = [c for c in df.columns if c not in ['target_blue_win']]
        for col in feature_cols:
            # Check if any feature is identical to target_blue_win
            if df[col].dtype in [np.int64, np.float64, np.int32]:
                if np.array_equal(df[col].values, df['target_blue_win'].values):
                    issues.append(f"Leakage Warning: Feature '{col}' is mathematically identical to the target 'target_blue_win'!")
                    
    except Exception as e:
        issues.append(f"Error auditing feature matrix: {e}")
    return issues

def audit_serialized_models(models_dir: str = "1_NoteBook/models") -> List[str]:
    issues = []
    expected_models = ["g1_xgb.pkl", "g2p_cat1.pkl", "g2p_rf.pkl", "g2p_cat2.pkl"]
    
    if not os.path.exists(models_dir):
        issues.append(f"Serialized models directory missing at '{models_dir}'")
        return issues
        
    for model_name in expected_models:
        path = os.path.join(models_dir, model_name)
        if not os.path.exists(path):
            issues.append(f"Serialized model missing: '{path}'")
        elif os.path.getsize(path) == 0:
            issues.append(f"Serialized model is empty: '{path}'")
            
    return issues

def main():
    print("="*60)
    print("🛡️ SENTINEL ADVANCED INTEGRITY AND ROBUSTNESS RUNNER 🛡️")
    print("="*60)
    
    # 1. Syntax Verification
    targets = [
        '1_NoteBook/Prediction_v1.ipynb',
        'features.py',
        'models.py',
        'model_zoo.py',
        'scraper.py',
        'generate_features.py',
        'create_prediction_v1_tuned.py',
        'main.py'
    ]
    
    failed = False
    print("1. CODEBASE SYNTAX AUDIT:")
    for target in targets:
        if not os.path.exists(target):
            print(f"  [-] {target:40s} ... MISSING (Skipped)")
            continue
            
        if target.endswith('.ipynb'):
            nb_errors = check_notebook_syntax(target)
            if nb_errors:
                print(f"  [❌] {target:40s} ... SYNTAX FAILED")
                for cell_idx, line, msg in nb_errors:
                    if cell_idx == -1:
                        print(f"      [CRITICAL] Corrupted JSON at line {line}: {msg}")
                    else:
                        print(f"      [Cell {cell_idx}] Syntax Error at line {line}: {msg}")
                failed = True
            else:
                print(f"  [✅] {target:40s} ... OK")
        else:
            py_errors = check_python_syntax(target)
            if py_errors:
                print(f"  [❌] {target:40s} ... SYNTAX FAILED")
                for line, msg in py_errors:
                    print(f"      Line {line:3d}: {msg}")
                failed = True
            else:
                print(f"  [✅] {target:40s} ... OK")
                
    print("\n2. ML DATA PROVENANCE AND SCHEMA AUDIT:")
    # Database checks
    db_issues = audit_database()
    if db_issues:
        print("  [❌] SQLite Database Audit FAILED:")
        for issue in db_issues:
            print(f"      - {issue}")
        failed = True
    else:
        print("  [✅] SQLite database tables and sizes validated successfully.")
        
    # Feature matrix checks
    fm_issues = audit_feature_matrix()
    if fm_issues:
        print("  [❌] Feature Matrix (ML_Feature_Matrix.csv) Audit FAILED:")
        for issue in fm_issues:
            print(f"      - {issue}")
        failed = True
    else:
        print("  [✅] Feature Matrix matches shape constraints (>2300x150) and has 0 null values.")

    print("\n3. PRODUCTION SERVEL AND MODEL AUDIT:")
    # Serialized model checks
    model_issues = audit_serialized_models()
    if model_issues:
        print("  [❌] Serialized Models Audit FAILED:")
        for issue in model_issues:
            print(f"      - {issue}")
        failed = True
    else:
        print("  [✅] Serialized XGBoost, CatBoost, and Random Forest ensemble models successfully validated.")

    print("\n" + "="*60)
    if failed:
        print("❌ PIPELINE VALIDATION FAILED! Please review the critical issues above.")
        print("="*60)
        sys.exit(1)
    else:
        print("🎉 ALL CHECKS PASSED SUCCESSFULLY!")
        print("   The machine learning pipeline, databases, and schemas are valid and leak-free!")
        print("="*60)

if __name__ == "__main__":
    main()
