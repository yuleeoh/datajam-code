# code was initially all written and executed via google colab, transported to github for ease of accessibility 
from google.colab import files
uploaded = files.upload() 

import pandas as pd

files = [
    "chronicdownload2018.xlsx",
    "chronicdownload2019.xlsx",
    "chronicdownload2022.xlsx",
    "chronicdownload2025.xlsx"
]

all_years = []

for file in files:
    df = pd.read_excel(file)

    df.columns = df.columns.str.lower().str.strip()

    df = df[[
        "countyname",
        "studentgroup",
        "currnumer",
        "currdenom"
    ]]

    df = df[df["studentgroup"] == "ALL"]

    df["currnumer"] = pd.to_numeric(df["currnumer"], errors="coerce")
    df["currdenom"] = pd.to_numeric(df["currdenom"], errors="coerce")

    df = df.dropna(subset=["countyname", "currnumer", "currdenom"])
    df = df[df["currdenom"] > 0]

    county_df = df.groupby("countyname").agg({
        "currnumer": "sum",
        "currdenom": "sum"
    }).reset_index()

    county_df["absenteeism_rate"] = (
        county_df["currnumer"] / county_df["currdenom"]
    )

    if "2018" in file or "2019" in file:
        county_df["period"] = "pre"
    elif "2022" in file:
        county_df["period"] = "early_post"
    elif "2025" in file:
        county_df["period"] = "late_post"

    all_years.append(county_df)

full_df = pd.concat(all_years, ignore_index=True)

summary = full_df.groupby(["countyname", "period"])["absenteeism_rate"].mean().reset_index()

pivot = summary.pivot(
    index="countyname",
    columns="period",
    values="absenteeism_rate"
).reset_index()

pivot = pivot.rename(columns={
    "pre": "pre_rate",
    "early_post": "early_post_rate",
    "late_post": "late_post_rate"
})

pivot = pivot.dropna()

pivot.to_excel("final_cleaned_data.xlsx", index=False)

pivot.head()

pivot_rounded = pivot.copy()

pivot_rounded["pre_rate"] = pivot_rounded["pre_rate"].round(3)
pivot_rounded["early_post_rate"] = pivot_rounded["early_post_rate"].round(3)
pivot_rounded["late_post_rate"] = pivot_rounded["late_post_rate"].round(3)

pivot_rounded.to_excel("final_cleaned_data.xlsx", index=False)

from google.colab import files
files.download("final_cleaned_data.xlsx")