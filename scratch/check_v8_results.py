import pandas as pd

# Load prediction tracker
df = pd.read_csv('csv_data/prediction_tracker.csv')

# Game 1 and Game 2+ splits
g1 = df[df['game_number'] == 1]
g2plus = df[df['game_number'] > 1]

# Calculate accuracies
acc_g1 = g1['correct'].mean()
acc_g2 = g2plus['correct'].mean()
acc_comb = df['correct'].mean()

print("=========================================")
print("🏆 VERIFIED V8 ENSEMBLE RESULTS:")
print(f"   Game 1 Accuracy  : {acc_g1 * 100:.2f}%")
print(f"   Game 2+ Accuracy : {acc_g2 * 100:.2f}%")
print(f"   📊 Combined Accuracy : {acc_comb * 100:.2f}%")
print("=========================================")
