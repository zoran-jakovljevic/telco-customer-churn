import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")

    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]:0, sorted_vals[1]:1}
        return s.astype(str).map(mapping).astype("Int64")

    return s