# Official MPL PH Stats Extract

Source: official MPL PH data pages at `https://ph-mpl.com/s{season}/data` and current `https://ph-mpl.com/data`.

Extracted seasons:

- Official data page available and extracted: Season 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17
- Not available from the same official data route: Season 1, 2, 3, 4
- Season 16 official archive route returned 404 during extraction: `https://ph-mpl.com/s16/data`

Files:

- `mpl_ph_official_players_s5_s15_s17.csv`
- `mpl_ph_official_teams_s5_s15_s17.csv`
- `mpl_ph_official_heroes_s5_s15_s17.csv`
- `mpl_ph_official_standings_s5_s15_s17.csv`
- `mpl_ph_official_stats_manifest.csv`

Schema note:

- Seasons 5-10, 14-15, and 17 expose player totals such as `total_kills`, `total_deaths`, `total_assists`, `kda_ratio`, and `kill_participation`.
- Seasons 11-13 expose average player stats such as `average_kills`, `average_deaths`, `average_assists`, `average_kda`, `average_damage_gold`, and `average_teamfight`. The official page does not provide the same total K/D/A columns for these seasons.
