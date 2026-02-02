# Qualtrics Data Analysis.py
# This script processes survey data from Qualtrics,
# specifically for GAD-7, PHQ-9, and WEMWBS questionnaires.
# To make data analysis easier, responses are recoded, columns renamed,
# and unnecessary rows and columns removed.
import pandas as pd
from datetime import datetime

# Load the data file
data_file = ""


# The excel file contains multiple sheets, all sheets are loaded into a dictionary of dataframes (dfs)
dfs = pd.read_excel(data_file, sheet_name=None)


for sheet_name, df in dfs.items():
    # Drop unnecessary columns
    dfs[sheet_name].drop(columns=["EndDate", "Finished", "ResponseId"], inplace=True)
    # Drop rows before xxxx-xx-xx as this was the first testing date (before only pilot data)
    mask = df["StartDate"].astype(str).str.startswith("xxxx-xx-xx")
    if mask.any():
        dfs[sheet_name] = df.loc[mask.idxmax() :].reset_index(drop=True)
    else:
        print("Something went wrong - Date not found")

# Rename columns for clarity
dfs["GAD-7"] = dfs["GAD-7"].rename(
    columns={
        "Q1": "ParticipantID",
        "Q2": "Trial Number",
        "Q3": "Q1",
        "Q4": "Q2",
        "Q5": "Q3",
        "Q6": "Q4",
        "Q7": "Q5",
        "Q8": "Q6",
        "Q9": "Q7",
    }
)
dfs["PHQ-9"] = dfs["PHQ-9"].rename(
    columns={
        "Q1": "ParticipantID",
        "Q2": "Trial Number",
        "Q3": "Q1",
        "Q4": "Q2",
        "Q5": "Q3",
        "Q6": "Q4",
        "Q7": "Q5",
        "Q8": "Q6",
        "Q9": "Q7",
        "Q10": "Q8",
        "Q11": "Q9",
    }
)
dfs["WEMWBS"] = dfs["WEMWBS"].rename(
    columns={
        "Q1": "ParticipantID",
        "Q2": "Trial Number",
        "Q3": "Q1",
        "Q4": "Q2",
        "Q5": "Q3",
        "Q6": "Q4",
        "Q7": "Q5",
        "Q8": "Q6",
        "Q9": "Q7",
        "Q10": "Q8",
        "Q11": "Q9",
        "Q12": "Q10",
        "Q13": "Q11",
        "Q14": "Q12",
        "Q15": "Q13",
        "Q16": "Q14",
    }
)
dfs["VAS"] = dfs["VAS"].rename(
    columns={
        "Q1": "ParticipantID",
        "Q2": "Trial Number",
        "Q3": "Time Point",
        "Q4_1": "Thermal Sensation",  # column names strange due to Qualtrics labeling
        "Q5_1": "Thermal Comfort",
        "Q7": "RPE",
    }
)
dfs["Hooper_ASBQ2"] = dfs["Hooper_ASBQ2"].rename(
    columns={
        "Q1.1": "ParticipantID",
        "Q1.2": "Trial Number",
        "Q1.3_1": "Sleep",
        "Q1.4_1": "Fatigue",
        "Q1.5_1": "Stress",
        "Q1.6_1": "Muscle Soreness",
    }
)

dfs["PSQI"] = dfs["PSQI"].rename(
    columns={
        "Q1": "ParticipantID",
        "Q2": "Trial Number",
    }
)


# Answers in Qualtrics are coded as 1-4, but GAD-7 and PHQ-9 scoring requires 0-3
# To make data analysis easier, answers are recoded
dfs["GAD-7"].loc[:, "Q1":"Q7"] = (
    dfs["GAD-7"].loc[:, "Q1":"Q7"].astype(int).replace({1: 0, 2: 1, 3: 2, 4: 3})
)
dfs["PHQ-9"].loc[:, "Q1":"Q9"] = (
    dfs["PHQ-9"].loc[:, "Q1":"Q9"].astype(int).replace({1: 0, 2: 1, 3: 2, 4: 3})
)


# Total Score columns for MH questionnaires
dfs["GAD-7"]["Total_Score"] = dfs["GAD-7"].loc[:, "Q1":"Q7"].sum(axis=1)
dfs["PHQ-9"]["Total_Score"] = dfs["PHQ-9"].loc[:, "Q1":"Q9"].sum(axis=1)
dfs["WEMWBS"]["Total_Score"] = (
    dfs["WEMWBS"].loc[:, "Q1":"Q14"].apply(pd.to_numeric, errors="coerce").sum(axis=1)
)

# Fix participant and trial number inconsistencies
# Trial numbers were not named consistenly and typos have been made for Participant IDs
# To enable further analysis trial numbers and particpant IDs are standardized
for sheet_name, df in dfs.items():

    if sheet_name != "VAS":
        dfs[sheet_name]["Trial Number"] = dfs[sheet_name]["Trial Number"].replace(
            {"1": "T1", "TN FAM": "TN"}
        )
    else:
        dfs[sheet_name]["Trial Number"].loc[
            dfs[sheet_name]["ParticipantID"].str.contains(r"HY(?=TN)")
        ] = "TN"

        dfs[sheet_name]["Time Point"] = dfs[sheet_name]["Time Point"].replace(
            {
                "1": 0,
                "2": 15,
                "3": 30,
                "4": 45,
                "5": 60,
            }
        )

    dfs[sheet_name]["ParticipantID"] = dfs[sheet_name]["ParticipantID"].replace(
        {
            "": "",
        }
    )


# Save cleaned data to new Excel/CSV file
timestamp = datetime.now().strftime("%Y%m%d")
combined_filename = f"combined_cleaned_qualtrics_data_{timestamp}.xlsx"
GAD_filename = f"GAD7_cleaned_{timestamp}.xlsx"
PHQ_filename = f"PHQ9_cleaned_{timestamp}.xlsx"
WEMWBS_filename = f"WEMWBS_cleaned_{timestamp}.xlsx"
# Save to one Excel file with multiple sheets
with pd.ExcelWriter(combined_filename, engine="openpyxl") as writer:
    for sheet_name, df in dfs.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
# Save into separate CSV files
dfs["GAD-7"].to_csv(f"GAD7_cleaned_{timestamp}.csv", index=False)
dfs["PHQ-9"].to_csv(f"PHQ9_cleaned_{timestamp}.csv", index=False)
dfs["WEMWBS"].to_csv(f"WEMWBS_cleaned_{timestamp}.csv", index=False)
dfs["VAS"].to_csv(f"VAS_cleaned_{timestamp}.csv", index=False)
