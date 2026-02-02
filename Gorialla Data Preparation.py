"""
Gorilla Cleaning Analysis

Processes experimental data exported from Gorilla,
for:
- Go/No-Go Task
- Digit Span Task
- Visual Search Task

Cleans participant IDs, separates tasks, computes task-specific
summary metrics, and exports results with timestamps.

Material page: ""
"""

import pandas as pd
from datetime import datetime

data_file = ""
df = pd.read_csv(data_file)


# --- Map Tree Node Keys to Pre and Post for each task for clarity ---
# Tree Node Keys are unique task IDS and can be found in the Gorilla experiment builder
df["Tree Node Key"] = df["Tree Node Key"].replace(
    {
        "task-do86": "Pre",
        "task-rv7f": "Post",
        "task-bxtx": "Pre",
        "task-t796": "Post",
        "task-vjv6": "Pre",
        "task-m599": "Post",
    }
)


# --- Rename Participant1 and Participant2 T1 rows ---
# During data collection (T1) with participants 1 and 2 a problem occured.
# There was a missmatch between cognitive tests the particpants started (pre) and finished (post)
# The problem was immediately documented.

# Drop empty rows (participants did not complete the tasks)
rows_to_remove = (
    (df["Participant Public ID"] == "") | (df["Participant Public ID"] == "")
) & (df["Tree Node Key"] == "Post")
df = df.loc[~rows_to_remove].reset_index(drop=True)

# "" T1 Pre and "" T1 Pre
df.loc[df["Tree Node Key"] == "Pre", "Participant Public ID"] = df.loc[
    df["Tree Node Key"] == "Pre", "Participant Public ID"
].replace(
    {
        "": " T1",
        " T1": " T1",
    }
)

# "" T1 Post
df.loc[df["Tree Node Key"] == "Post", "Participant Public ID"] = df.loc[
    df["Tree Node Key"] == "Post", "Participant Public ID"
].replace(
    {
        " T1": " T1",
    }
)

# "" T1 Post
# Only get the row where Participant Public ID is "" and Tree Node Key is Pre
mask_p2_post = (df["Participant Public ID"] == "") & (df["Tree Node Key"] == "Pre")

# Change values
df.loc[mask_p2_post, "Participant Public ID"] = " T1"
df.loc[mask_p2_post, "Tree Node Key"] = "Post"


# --- Clean Participant Public IDs and extract Session Number ---
df["Session Number"] = (
    df["Participant Public ID"].str.extract(r"(T\d+)$").fillna("TN")
)  # TN = Thermoneutral
df["Participant Public ID"] = df["Participant Public ID"].str.replace(
    r" T\d+$", "", regex=True
)
df["Participant Public ID"] = df["Participant Public ID"].replace({"": "", "": ""})


# --- Separate dataframes for each task ---
df1 = df[df["Task Name"] == "Go_No_Go_Task"]
df2 = df[df["Task Name"].str.contains("random digit span task", case=False, na=False)]
df3 = df[
    df["Task Name"].str.contains("Visual Search Task")
    & (df["Object Name"] == "Keyboard responses")
]


# --- Process Go/No-Go Task Data ---
# Total number of Go/No Go trials
df1["is_go"] = df1["Answer"] == "Go"
df1["is_nogo"] = df1["Answer"] == "No Go"
# Correct Go responses
df1["go_correct"] = df1["is_go"] & (df1["Response"] == "Go")
# Incorrect No Go responses
df1["nogo_incorrect"] = df1["is_nogo"] & (df1["Response"] == "Go")
# Mean reaction times for Go trials
df1["rt_go"] = df1["Reaction Time"].where(df1["is_go"])


gonogo_result = (
    df1.groupby(["Participant Public ID", "Tree Node Key", "Session Number"])
    .agg(
        n_go_correct=("go_correct", "sum"),
        n_nogo_incorrect=("nogo_incorrect", "sum"),
        mean_rt_go=("rt_go", "mean"),
    )
    .reset_index()
)
# Calculate accuracy
gonogo_result["Accuracy"] = (
    (gonogo_result["n_go_correct"] + (8 - gonogo_result["n_nogo_incorrect"])) / 40 * 100
)


# --- Process Digit Span Task Data ---
# one row per participant, with max digit span achieved
digit_span_result = (
    df2.groupby(["Participant Public ID", "Tree Node Key", "Session Number"])
    .agg(digit_span=("Max Digit Span", "max"))
    .reset_index()
)


# --- Process Visual Search Task Data ---
# one row per participant, with columns for each trial's correctness and reaction time
df3_wide = df3.pivot(
    index=["Participant Public ID", "Tree Node Key", "Session Number"],
    columns="Trial Number.2",
    values=["Correct", "Reaction Time.1"],
).reset_index()

# Rename columns for clarity (E.g., Correct 1, Reaction Time 1, etc.)
df3_wide.columns = [
    f"{'Reaction Time' if 'Reaction Time.1' in col[0] else col[0]} {col[1]}"
    for col in df3_wide.columns
]
visual_search_result = df3_wide.fillna(0)  # to have (1 or) 0 instead of "missing value"


# --- Export cleaned results with timestamp ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
gonogo_filename = f"go_nogo_results_{timestamp}"
digit_span_filename = f"digit_span_results_{timestamp}"
visual_search_filename = f"visual_search_results_{timestamp}"
combined_filename = f"combined_cleaned_cognitive_task_data_{timestamp}.xlsx"
# Save to CSV
gonogo_result.to_csv(gonogo_filename + ".csv", index=False)
digit_span_result.to_csv(digit_span_filename + ".csv", index=False)
visual_search_result.to_csv(visual_search_filename + ".csv", index=False)
# Save to one Excel file with multiple sheets
with pd.ExcelWriter(combined_filename, engine="openpyxl") as writer:
    gonogo_result.to_excel(writer, sheet_name="Go-NoGo", index=False)
    digit_span_result.to_excel(writer, sheet_name="Digit Span", index=False)
    visual_search_result.to_excel(writer, sheet_name="Visual Search", index=False)
