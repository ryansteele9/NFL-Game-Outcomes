
### DEPRECATED: replaced by download_nflverse.py (Python port, same logic).
### Downloads play-by-play EPA statistics from nflfastR and computes EPA 
### related metrics. Groups play-by-play data into data for each game and joins all
### data into one big file for all games in all seasons (2022-2025). 
### File saved to: data/external/nflfastr/team_game_advanced_2022_2025.csv

library(dplyr)
library(nflfastR)
library(nflreadr)
library(readr)

seasons <- 2022:2025

output_path <- "data/external/nflfastr/team_game_advanced_2022_2025.csv"

pbp <- nflreadr::load_pbp(seasons)

pbp <- pbp %>% filter(season_type == "REG")

# Offensive stats
off_game <- pbp %>%
filter(!is.na(epa), !is.na(posteam)) %>%
group_by(season, week, team = posteam, game_id) %>%
summarise(
    off_plays = n(),
    off_epa_per_play = mean(epa, na.rm = TRUE),
    off_success_rate = mean(epa > 0, na.rm = TRUE),

    off_dropbacks = sum(pass == 1| qb_scramble == 1, na.rm = TRUE),
    off_dropback_epa = ifelse(
        off_dropbacks > 0,
        mean(epa[pass == 1 | qb_scramble == 1], na.rm = TRUE),
        NA_real_
    ),

    off_rushes= sum(rush == 1, na.rm = TRUE),
    off_rush_epa = ifelse(
        off_rushes > 1,
        mean(epa[rush == 1], na.rm = TRUE),
        NA_real_
    ),

    .groups= "drop"
)

# Defensive stats
def_game <- pbp%>%
filter(!is.na(epa), !is.na(defteam)) %>%
group_by(season, week, team = defteam, game_id) %>%
summarise(
    def_plays = n(),
    def_epa_per_play = mean(epa, na.rm = TRUE),
    def_success_rate = mean(epa > 0, na.rm = TRUE),

    def_dropbacks_against = sum(pass == 1 | qb_scramble == 1, na.rm = TRUE),
    def_dropback_epa_against = ifelse(
        def_dropbacks_against > 0,
        mean(epa[pass == 1 | qb_scramble == 1], na.rm = TRUE),
        NA_real_
    ),

    def_rushes_against = sum(rush == 1, na.rm = TRUE),
    def_rush_epa_against = ifelse(
        def_rushes_against > 0,
        mean(epa[rush == 1], na.rm = TRUE),
        NA_real_
    ),

    .groups = "drop"
)

# Combine offensive and defensive stats
team_game_epa <- off_game %>%
full_join(def_game, by = c("season", "week", "team", "game_id"))

dir.create("data/external/nflfastr", recursive = TRUE, showWarnings = FALSE)

write_csv(team_game_epa, output_path)

cat("Saved team-game EPA metrics to: ", output_path, "\n")