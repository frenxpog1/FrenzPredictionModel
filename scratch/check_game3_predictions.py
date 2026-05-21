import pandas as pd
import numpy as np

# Load the prediction tracker that was saved during the copy_code.py execution
try:
    df_tracker = pd.read_csv('csv_data/prediction_tracker.csv')
    print("Loaded prediction tracker successfully!")
except Exception as e:
    print("Error loading prediction tracker:", e)
    exit(1)

# Filter for Game 3 matches
df_g3 = df_tracker[df_tracker['game_number'] == 3].copy()
print(f"Total Game 3 matches in test set: {len(df_g3)}")

if len(df_g3) > 0:
    # Check how many times Team 2 (red_side_team if t1 won, or t2?)
    # Wait, let's see who is predicted as winner in Game 3
    # Let's count actual and predicted winners for Game 3
    correct_count = df_g3['correct'].sum()
    print(f"Game 3 Accuracy in Test Set: {correct_count / len(df_g3) * 100:.2f}% ({correct_count}/{len(df_g3)})")
    
    # Check if actual winner is always red_side_team
    # Let's check how many times actual winner equals red_side_team
    actual_red = (df_g3['actual_winner'] == df_g3['red_side_team']).sum()
    print(f"Actual winner is RED side team in Game 3: {actual_red}/{len(df_g3)} ({actual_red/len(df_g3)*100:.1f}%)")
    
    # Check if predicted winner is always red_side_team
    pred_red = (df_g3['predicted_winner'] == df_g3['red_side_team']).sum()
    print(f"Predicted winner is RED side team in Game 3: {pred_red}/{len(df_g3)} ({pred_red/len(df_g3)*100:.1f}%)")
    
    print("\nSample Game 3 Predictions:")
    print(df_g3[['season', 'blue_side_team', 'red_side_team', 'predicted_winner', 'actual_winner', 'confidence', 'correct']].head(10))
else:
    print("No Game 3 matches found in the test set!")
