import create_prediction_v1_tuned as p

for name in ['base_features', 'g1_features', 'g2_features', 'g2plus_features', 'g3plus_features']:
    lst = getattr(p, name, [])
    dupes = [x for x in lst if lst.count(x) > 1]
    if dupes:
        print(f'{name} duplicates:', set(dupes))
