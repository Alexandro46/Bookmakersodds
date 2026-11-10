# Football betting market data

This repository contains a downloader for the Football-Data.co.uk match CSVs used in the proposal's analysis of bookmaker calibration, favourite–longshot bias, differences across leagues, and possible gains from adding historical team form.

The script retrieves the 2019/20 through 2024/25 seasons for the first and second divisions in England, Germany, Italy, Spain, and France. It downloads the publisher's files unchanged; it does not clean odds or calculate probabilities.

## Run it on Windows

1. Install Python 3 if it is not already installed. During installation, enable **Add Python to PATH**.
2. Open PowerShell and copy these commands:

```powershell
git clone https://github.com/Alexandro46/Bookmakersodds.git
cd Bookmakersodds
py download_data.py
```

If your computer uses `python` instead of the Windows `py` launcher, run:

```powershell
python download_data.py
```

No extra Python packages are needed. The script uses Python's standard library.

## Output

CSV files are saved in `data/football_data/raw/<season>/`, with one file per division. For example:

```text
data/football_data/raw/
  1920/
    E0.csv
    E1.csv
    D1.csv
    D2.csv
    I1.csv
    I2.csv
    SP1.csv
    SP2.csv
    F1.csv
    F2.csv
  ...
  2425/
  manifest.json
```

The six season folders each contain ten league files (60 files in total if Football-Data has every requested file available). The manifest records the source URL and whether each file was downloaded, skipped because it already existed, or unavailable. Existing files are skipped so repeat runs are safe. To fetch them again and replace the local copies:

```powershell
py download_data.py --overwrite
```

The script validates that a response looks like a match CSV, retries temporary network errors, and reports unavailable files without stopping the rest of the downloads. A non-zero exit code means one or more files need review in the manifest or a later retry.

## Season and division codes

| Country | First division | Second division |
| --- | --- | --- |
| England | Premier League (`E0`) | Championship (`E1`) |
| Germany | Bundesliga (`D1`) | 2. Bundesliga (`D2`) |
| Italy | Serie A (`I1`) | Serie B (`I2`) |
| Spain | La Liga (`SP1`) | Segunda División (`SP2`) |
| France | Ligue 1 (`F1`) | Ligue 2 (`F2`) |

Season folder `1920` represents 2019/20; `2425` represents 2024/25.

## How this supports the project

Start by checking CSV headers and missingness across seasons and divisions. Football-Data indicates that from 2019/20 it provides opening and closing odds; closing-odds columns use a `C` in their names. Confirm the exact closing-odds bookmaker columns present in each file before choosing a common sample. Then harmonise team names and results, remove the bookmaker margin to create implied probabilities, and calculate calibration metrics. Historical form features should be constructed using only matches played before each prediction date, with time-ordered validation to avoid look-ahead bias.

Source: [Football-Data.co.uk European data downloads](https://www.football-data.co.uk/downloadm.php). The availability and columns in historical CSVs can vary, so inspect the downloaded files before analysis.
