from pathlib import Path

import pandas as pd


def load_3le_raw(path: Path) -> pd.DataFrame:
    """Load a Space-Track 3-line element file into a DataFrame."""
    rows = []
    lines = path.read_text().splitlines()

    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break

        name_line = lines[i].strip()
        name = name_line[2:].strip() if name_line.startswith("0 ") else name_line

        rows.append(
            {
                "name": name,
                "line1": lines[i + 1].strip(),
                "line2": lines[i + 2].strip(),
            }
        )

    return pd.DataFrame(rows)


def spacetrack_parse_file(path: Path) -> pd.DataFrame:
    """Parse a 3-line element file and enrich with Keplerian elements."""
    from orbx.clustering.data_handling.DataHandler import DataHandler

    return DataHandler().tle_to_keplerian(load_3le_raw(path))
