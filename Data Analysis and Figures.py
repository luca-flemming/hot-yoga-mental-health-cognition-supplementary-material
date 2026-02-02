import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# --- Load Data and Prepare
# Cognitive Tasks
gng_df = pd.read_csv("")
ds_df = pd.read_csv("")
vs_df = pd.read_csv("")

# Adjust vs_df for analysis
vs_df.rename(
    columns={
        "Participant Public ID ": "Participant Public ID",
        "Tree Node Key ": "Tree Node Key",
        "Session Number ": "Session Number",
    },
    inplace=True,
)
correct_cols = [col for col in vs_df.columns if col.startswith("Correct")]
rt_cols = [col for col in vs_df.columns if col.startswith("Reaction Time")]
vs_df["Accuracy"] = vs_df[correct_cols].sum(axis=1) / 8 * 100
vs_df["mean_rt"] = vs_df[rt_cols].mean(axis=1)


# Only keep HY participants in the data set (exclude if participant ID contains TN after PW)
gng_hy_df = gng_df[~(gng_df["Participant Public ID"].str.contains(r"PW(?=TN)"))]
ds_hy_df = ds_df[~(ds_df["Participant Public ID"].str.contains(r"PW(?=TN)"))]
vs_hy_df = vs_df[~(vs_df["Participant Public ID"].str.contains(r"PW(?=TN)"))]


# Mental Health
gad7_df = pd.read_csv("")
phq9_df = pd.read_csv("")
wb_df = pd.read_csv("")


# Only keep HY participants (exclude if participant ID contains TN after PW)
gad7_hy_df = gad7_df[~(gad7_df["ParticipantID"].str.contains(r"PW(?=TN)"))]
phq9_hy_df = phq9_df[~(phq9_df["ParticipantID"].str.contains(r"PW(?=TN)"))]
wb_hy_df = wb_df[~(wb_df["ParticipantID"].str.contains(r"PW(?=TN)"))]

# LOAD FILES
temp_hr_df = pd.read_csv("")
vas_df = pd.read_csv("")
# ADJUST vas_df
vas_hy_df = vas_df[~(vas_df["ParticipantID"].str.contains(r"PW(?=TN)"))]

# Fix data collection mistake and format
vas_hy_df.loc[vas_hy_df["StartDate"] == "2025-12-04 11:35:53"] = vas_hy_df.loc[
    vas_hy_df["StartDate"] == "2025-12-04 11:35:53"
].replace({"": ""})

vas_hy_df["Trial Number"] = vas_hy_df["Trial Number"].replace(
    {"1": "T1", "2": "T2", "3": "T3", "4": "T4", "5": "T5"}
)
vas_hy_df["RPE"] = vas_hy_df["RPE"].replace(
    {
        1: 6,
        2: 7,
        3: 8,
        4: 9,
        5: 10,
        6: 11,
        7: 12,
        8: 13,
        9: 14,
        10: 15,
        11: 16,
        12: 17,
        13: 18,
        14: 19,
        15: 20,
    }
)


# Order Trial Number and Time Point
def order_df(df):
    df["Time Point"] = pd.Categorical(
        df["Time Point"],
        categories=[0, 15, 30, 45, 60],
        ordered=True,
    )
    df["Trial Number"] = pd.Categorical(
        df["Trial Number"],
        categories=["TN", "T1", "T2", "T3", "T4", "T5"],
        ordered=True,
    )

    return df


temp_hr_df = order_df(temp_hr_df)
vas_hy_df = order_df(vas_hy_df)

numeric_cols = ["Core T", "Heart Rate"]
temp_hr_df[numeric_cols] = temp_hr_df[numeric_cols].replace(r"[^\d.]", "", regex=True)
temp_hr_df[numeric_cols] = temp_hr_df[numeric_cols].apply(
    pd.to_numeric, errors="coerce"
)

# Hooper Scale and Urine Osmolality
hooper_urine_df = pd.read_csv("")

trial_cols = ["TN", "T1", "T2", "T3", "T4", "T5"]

hooper_urine_df = hooper_urine_df.melt(
    id_vars=["ParticipantID", "Condition"],
    value_vars=trial_cols,
    var_name="Trial Number",
    value_name="Value",
)

hooper_urine_df = hooper_urine_df.pivot(
    index=["ParticipantID", "Trial Number"],
    columns="Condition",
    values="Value",
).reset_index()


# --- Create Figure Functions---
def plot_session_prepost_metric(
    df,
    metric_col,
    session_col="Session Number",
    condition_col="Tree Node Key",
    participant_col="Participant Public ID",
    y_label=None,
    y_lim=None,
    spread=0.12,
    figsize=(12, 6),
    # figure_title=None,
    figure_file_name=None,
    panel_text=None,
):
    """
    Plot group means (barplot) with individual participant trajectories
    across sessions for a given metric (e.g. Accuracy, Reaction Time).
    """

    # Data Preparation
    plot_df = df.copy()

    # Session & condition maps
    condition_order = ("Pre", "Post")
    session_order = ["TN", "T1", "T2", "T3", "T4", "T5"]
    session_map = {s: i for i, s in enumerate(session_order)}
    condition_offset = {"Pre": -0.2, "Post": 0.2}
    condition_sort = {"Pre": 0, "Post": 1}

    # Jitter calculation
    collision_keys = [session_col, condition_col, metric_col]

    count_same = plot_df.groupby(collision_keys)[metric_col].transform("count")
    ranks = plot_df.groupby(collision_keys).cumcount()

    plot_df["jitter"] = np.where(
        count_same > 1,
        (ranks - (count_same - 1) / 2) * spread,
        0,
    )

    plot_df["x_coords"] = (
        plot_df[session_col].map(session_map).astype(float)
        + plot_df[condition_col].map(condition_offset)
        + plot_df["jitter"]
    )

    # Participant colours
    unique_ids = sorted(plot_df[participant_col].unique())
    palette = sns.color_palette("husl", len(unique_ids))
    color_map = dict(zip(unique_ids, palette))

    marker_map = {"P1": "o", "P2": "s"}

    # Plot setup
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=figsize)

    sns.barplot(
        data=plot_df,
        x=session_col,
        y=metric_col,
        hue=condition_col,
        order=session_order,
        hue_order=condition_order,
        errorbar="sd",
        capsize=0.15,
        palette="colorblind",
        alpha=0.7,
        zorder=1,
    )

    # Individual trajectories
    for pid, group in plot_df.groupby(participant_col):

        group = group.sort_values(
            [session_col, condition_col],
            key=lambda x: (
                x.map(session_map) if x.name == session_col else x.map(condition_sort)
            ),
        )

        color = color_map[pid]
        marker = marker_map.get(pid[:2], "x")

        plt.scatter(
            group["x_coords"],
            group[metric_col],
            color=color,
            marker=marker,
            s=60,
            edgecolor="black",
            linewidth=1,
            zorder=3,
        )

        plt.plot(
            group["x_coords"],
            group[metric_col],
            color=color,
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )

    # Legend & formatting
    handles, labels = plt.gca().get_legend_handles_labels()

    participant_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=palette[0],
            label="P1",
            markerfacecolor=palette[0],
            markeredgecolor="black",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color=palette[1],
            label="P2",
            markerfacecolor=palette[1],
            markeredgecolor="black",
            markersize=10,
        ),
    ]

    plt.legend(
        handles=handles[: len(condition_order)] + participant_legend,
        loc="upper left",
        bbox_to_anchor=(1.05, 0.95),
    )

    plt.ylabel(y_label or metric_col)
    plt.xlabel("Trial Number")

    fig = plt.gcf()  # get current figure

    if panel_text is not None:
        fig.text(
            -0.025,
            0.95,  # x, y in figure coordinates
            panel_text,
            fontsize=25,
            fontweight="bold",
            va="top",
            ha="left",
        )

    if y_lim is not None:
        plt.ylim(y_lim)

    sns.despine(trim=True)
    plt.tight_layout()

    if figure_file_name is not None:
        plt.savefig(
            figure_file_name,
            dpi=600,
            format=figure_file_name.split(".")[-1],
            bbox_inches="tight",
        )

    plt.show()


def plot_mh_session_metric(
    df,
    metric_col,
    session_col="Trial Number",
    participant_col="ParticipantID",
    y_label=None,
    y_lim=None,
    spread=0.12,
    figsize=(12, 6),
    # figure_title=None,
    figure_file_name=None,
    panel_text=None,
):
    """
    Plot group means (barplot) with individual participant trajectories
    across sessions for a given metric (e.g. Accuracy, Reaction Time),
    with jitter to avoid overlapping dots.
    """

    plot_df = df.copy()

    # ----------------------
    # Define session order & numeric mapping
    # ----------------------
    session_order = ["TN", "T1", "T2", "T3", "T4", "T5"]
    session_map = {s: i for i, s in enumerate(session_order)}
    plot_df["_session_num"] = plot_df[session_col].map(session_map)
    # =========================
    # Jitter calculation
    # =========================
    collision_keys = [session_col, metric_col]

    count_same = plot_df.groupby(collision_keys)[metric_col].transform("count")
    ranks = plot_df.groupby(collision_keys).cumcount()

    plot_df["jitter"] = np.where(
        count_same > 1,
        (ranks - (count_same - 1) / 2) * spread,
        0,
    )

    plot_df["x_coords"] = (
        plot_df[session_col].map(session_map).astype(float) + plot_df["jitter"]
    )

    # ----------------------
    # Participant colors and markers
    # ----------------------
    unique_ids = sorted(plot_df[participant_col].unique())
    palette = sns.color_palette("colorblind")[1:3]
    color_map = dict(zip(unique_ids, palette))
    marker_map = {"P1": "o", "P2": "s"}  # default markers

    # ----------------------
    # Plot setup
    # ----------------------
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=figsize)

    # Main barplot (group mean)
    sns.barplot(
        data=plot_df,
        x=session_col,
        y=metric_col,
        order=session_order,
        errorbar="sd",
        capsize=0.15,
        color=sns.color_palette("colorblind")[0],
        alpha=0.7,
        zorder=1,
    )

    # ----------------------
    # Plot individual trajectories
    # ----------------------
    for pid, group in plot_df.groupby(participant_col):

        group = group.sort_values("_session_num")  # ensures line moves left → right
        x = group["x_coords"].values
        y = group[metric_col].values
        color = color_map[pid]
        marker = marker_map.get(pid[:2], "x")

        plt.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=60,
            edgecolor="black",
            linewidth=1,
            zorder=3,
        )

        plt.plot(x, y, color=color, linestyle="--", linewidth=1.5, zorder=2)

    # ----------------------
    # Legend
    # ----------------------
    participant_legend = [
        Line2D(
            [0],
            [0],
            marker=marker_map.get(pid[:2], "x"),
            color=color_map[pid],
            label=pid[:2],
            markerfacecolor=color_map[pid],
            markeredgecolor="black",
            markersize=10,
        )
        for pid in unique_ids
    ]

    plt.legend(
        handles=participant_legend,
        loc="upper left",
        bbox_to_anchor=(1.05, 0.95),
    )

    fig = plt.gcf()  # get current figure

    if panel_text is not None:
        fig.text(
            -0.025,
            0.95,  # x, y in figure coordinates
            panel_text,
            fontsize=25,
            fontweight="bold",
            va="top",
            ha="left",
        )

    plt.ylabel(y_label or metric_col)
    if y_lim is not None:
        plt.ylim(y_lim)

    if figure_file_name == "GAD7_graph.png":
        plt.yticks(np.arange(0, 21, 5))
    # if figure_title is not None:
    #     plt.title(figure_title, pad=10, fontsize=20, fontweight="bold")

    sns.despine()
    plt.tight_layout()

    if figure_file_name is not None:
        plt.savefig(
            figure_file_name,
            dpi=600,
            format=figure_file_name.split(".")[-1],
            bbox_inches="tight",
        )

    plt.show()


def plot_within_session_metric(
    df,
    metric_col,
    participant_col="ParticipantID",
    time_col="Time Point",
    hue_col="Trial Number",
    x_label=None,
    y_label=None,
    y_lim=None,
    # title=None,
    figsize=(6, 10),
    save_path=None,
    panel_text=False,
):

    plot_df = df.copy()

    # ----------------------
    # Session order & sorting
    # ----------------------
    session_order = ["TN", "T1", "T2", "T3", "T4", "T5"]
    session_map = {s: i for i, s in enumerate(session_order)}

    plot_df["_session_num"] = plot_df[hue_col].map(session_map)
    plot_df = plot_df.sort_values([participant_col, "_session_num", time_col])

    participants = plot_df[participant_col].unique()
    n = len(participants)

    # ----------------------
    # Grayscale palette (TN → black, T5 → light gray)
    # ----------------------
    gray_levels = np.linspace(0.1, 0.75, len(session_order))
    color_map = dict(zip(session_order, [(g, g, g) for g in gray_levels]))

    # Line styles (optional but preserved)
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    line_style_map = dict(zip(session_order, line_styles))

    # ----------------------
    # Plot setup
    # ----------------------
    sns.set_theme(style="whitegrid", context="talk")
    panel_letter = ["A", "B"]

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(figsize[0] * n, figsize[1]),
    )

    if n == 1:
        axes = [axes]

    # ----------------------
    # Plot
    # ----------------------
    for ax, pid in zip(axes, participants):

        sub_df = plot_df[plot_df[participant_col] == pid]

        for sess in session_order:
            sess_df = sub_df[sub_df[hue_col] == sess]
            if sess_df.empty:
                continue

            ax.plot(
                sess_df[time_col],
                sess_df[metric_col],
                color=color_map[sess],
                linestyle=line_style_map[sess],
                linewidth=2,
                marker="o",
                markersize=6,
                label=sess,
            )

        # if y_label is not None:
        #     ax.set_title(
        #         f"{y_label}, {pid[:2]}", fontsize=20, fontweight="bold", y=1.05
        #     )
        # else:
        #     ax.set_title(f"Participant {pid[:2]}", fontsize=20, fontweight="bold")
        ax.set_xlabel(x_label or time_col)
        ax.set_ylabel(y_label or metric_col)
        ax.set_xticks(df[time_col].cat.categories)
        ax.set_xticklabels(df[time_col].cat.categories)

        if panel_text is not False:
            ax.text(
                -0.14,
                1.00,
                panel_letter[int(pid[1:2]) - 1],
                transform=ax.transAxes,
                fontsize=25,
                fontweight="bold",
                va="top",
                ha="left",
            )

        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.grid(True, linestyle="--", alpha=0.3)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 0.95),
        )

    sns.despine(trim=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()


# --- Create ANOVA Result Formatting Function ---
def format_rm_anova(aov, task, measure):
    if "ddof1" in aov.columns and "ddof2" in aov.columns:
        cols = [
            "Source",
            "ddof1",
            "ddof2",
            "F",
            "p-unc",
            "p-GG-corr",
            "ng2",
            "eps",
        ]
    elif "DF" in aov.columns:
        cols = [
            "Source",
            "DF",
            "F",
            "p-unc",
            "p-GG-corr",
            "ng2",
            "eps",
        ]

    out = (
        aov[cols]
        .assign(Task=task, Measure=measure)
        .rename(columns={"ng2": "eta2-G"})
        .round({"F": 2, "p-unc": 4, "p-GG-corr": 4, "eta2-G": 3})
    )

    return out


# --- Plot Figures ---
# GONOGO TASK GRAPH
plot_session_prepost_metric(
    df=gng_hy_df,
    metric_col="Accuracy",
    y_lim=(80, 102),
    y_label="Accuracy (%)",
    figure_file_name="Go_NoGo_Acc.png",
    panel_text="A",
)
plot_session_prepost_metric(
    df=gng_hy_df,
    metric_col="mean_rt_go",
    y_lim=(0, 510),
    y_label="Mean Reaction Time (ms)",
    figure_file_name="Go_NoGo_RT.png",
    panel_text="B",
)

# DIGIT SPAN TASK GRAPH
plot_session_prepost_metric(
    df=ds_hy_df,
    metric_col="digit_span",
    y_label="Maximum Digit Span",
    figure_file_name="Digit_Span.png",
)

# VISUAL SEARCH TASK GRAPH
plot_session_prepost_metric(
    df=vs_hy_df,
    metric_col="Accuracy",
    y_label="Accuracy (%)",
    y_lim=((60, 102)),
    figure_file_name="VS_Acc.png",
    panel_text="A",
)
plot_session_prepost_metric(
    df=vs_hy_df,
    metric_col="mean_rt",
    y_label="Mean Reaction Time (ms)",
    figure_file_name="VS_RT.png",
    panel_text="B",
)

# GAD7 GRAPH
plot_mh_session_metric(
    df=gad7_hy_df,
    metric_col="Total_Score",
    y_lim=(0, 20),
    y_label="GAD-7 Score (A.U.)",
    figure_file_name="GAD7_graph.png",
    panel_text="A",
)

# PHQ9 GRAPH
plot_mh_session_metric(
    df=phq9_hy_df,
    metric_col="Total_Score",
    y_lim=(0, 27),
    y_label="PHQ-9 Score (A.U.)",
    figure_file_name="PHQ9_graph.png",
    panel_text="B",
)

# WB GRAPH
plot_mh_session_metric(
    df=wb_hy_df,
    metric_col="Total_Score",
    y_lim=(0, 70),
    y_label="WEMWBS Score (A.U.)",
    figure_file_name="WEMWBS_graph.png",
    panel_text="C",
)

# Core Temperature
plot_within_session_metric(
    temp_hr_df,
    "Core T",
    save_path="Core_Temp.png",
    y_lim=(36.7, 38),
    y_label="Core Temperature (°C)",
    x_label="Time Point (min)",
    panel_text=True,
)

# Heart Rate
plot_within_session_metric(
    temp_hr_df,
    "Heart Rate",
    save_path="Heart_Rate.png",
    y_label="Heart Rate (BPM)",
    x_label="Time Point (min)",
    y_lim=(50, 145),
    panel_text=True,
)

# Thermal Sensation
plot_within_session_metric(
    vas_hy_df,
    "Thermal Sensation",
    save_path="Thermal_Sensation.png",
    y_lim=(0, 100),
    y_label="Thermal Sensation (A.U.)",
    x_label=("Time Point (min)"),
    panel_text=True,
)

# Thermal Comfort
plot_within_session_metric(
    vas_hy_df,
    "Thermal Comfort",
    save_path="Thermal_Comfort.png",
    y_lim=(0, 100),
    y_label="Thermal Comfort (A.U.)",
    x_label=("Time Point (min)"),
    panel_text=True,
)

# RPE
plot_within_session_metric(
    vas_hy_df,
    "RPE",
    save_path="RPE.png",
    y_lim=(6, 20),
    y_label="RPE (A.U.)",
    x_label=("Time Point (min)"),
    panel_text=True,
)


# --- Calculate ANOVAs ---

gng_acc_aov = pg.rm_anova(
    dv="Accuracy",
    within=["Session Number", "Tree Node Key"],
    subject="Participant Public ID",
    data=gng_hy_df,
    detailed=True,
    correction=True,
)

gng_rct_aov = pg.rm_anova(
    dv="mean_rt_go",
    within=["Session Number", "Tree Node Key"],
    subject="Participant Public ID",
    data=gng_hy_df,
    detailed=True,
    correction=True,
)

ds_ds_aov = pg.rm_anova(
    dv="digit_span",
    within=["Session Number", "Tree Node Key"],
    subject="Participant Public ID",
    data=ds_hy_df,
    detailed=True,
    correction=True,
)

vs_acc_aov = pg.rm_anova(
    dv="Accuracy",
    within=["Session Number", "Tree Node Key"],
    subject="Participant Public ID",
    data=vs_hy_df,
    detailed=True,
    correction=True,
)

vs_rct_aov = pg.rm_anova(
    dv="mean_rt",
    within=["Session Number", "Tree Node Key"],
    subject="Participant Public ID",
    data=vs_hy_df,
    detailed=True,
    correction=True,
)

# ANOVAS Mental Health QUESTIONNAIRES

gad7_aov = pg.rm_anova(
    dv="Total_Score",
    within=["Trial Number"],
    subject="ParticipantID",
    data=gad7_hy_df,
    detailed=True,
    correction=True,
)

phq9_aov = pg.rm_anova(
    dv="Total_Score",
    within=["Trial Number"],
    subject="ParticipantID",
    data=phq9_hy_df,
    detailed=True,
    correction=True,
)

wb_aov = pg.rm_anova(
    dv="Total_Score",
    within=["Trial Number"],
    subject="ParticipantID",
    data=wb_hy_df,
    detailed=True,
    correction=True,
)

# Core Temperature
core_t_aov = pg.rm_anova(
    dv="Core T",
    within=["Trial Number", "Time Point"],
    subject="ParticipantID",
    data=temp_hr_df,
    detailed=True,
    correction=True,
)

# Heart Rate
hr_aov = pg.rm_anova(
    dv="Heart Rate",
    within=["Trial Number", "Time Point"],
    subject="ParticipantID",
    data=temp_hr_df,
    detailed=True,
    correction=True,
)

# Perceptual Measurements
tsen_aov = pg.rm_anova(
    dv="Thermal Sensation",
    within=["Trial Number", "Time Point"],
    subject="ParticipantID",
    data=vas_hy_df.loc[(vas_hy_df["Trial Number"] != "T5")],
    detailed=True,
    correction=True,
)

tcom_aov = pg.rm_anova(
    dv="Thermal Comfort",
    within=["Trial Number", "Time Point"],
    subject="ParticipantID",
    data=vas_hy_df.loc[(vas_hy_df["Trial Number"] != "T5")],
    detailed=True,
    correction=True,
)

rpe_aov = pg.rm_anova(
    dv="RPE",
    within=["Trial Number", "Time Point"],
    subject="ParticipantID",
    data=vas_hy_df.loc[(vas_hy_df["Trial Number"] != "T5")],
    detailed=True,
    correction=True,
)

# Hooper Scale
hooper_sleep_aov = pg.rm_anova(
    dv="Sleep",
    within=["Trial Number"],
    subject="ParticipantID",
    data=hooper_urine_df,
    detailed=True,
    correction=True,
)

hooper_fatigue_aov = pg.rm_anova(
    dv="Fatigue",
    within=["Trial Number"],
    subject="ParticipantID",
    data=hooper_urine_df,
    detailed=True,
    correction=True,
)
hooper_ms_aov = pg.rm_anova(
    dv="Muscle Soreness",
    within=["Trial Number"],
    subject="ParticipantID",
    data=hooper_urine_df,
    detailed=True,
    correction=True,
)
hooper_stress_aov = pg.rm_anova(
    dv="Stress",
    within=["Trial Number"],
    subject="ParticipantID",
    data=hooper_urine_df,
    detailed=True,
    correction=True,
)

# Urine Osmolality
hooper_urine_aov = pg.rm_anova(
    dv="Urine Osmolality",
    within=["Trial Number"],
    subject="ParticipantID",
    data=hooper_urine_df,
    detailed=True,
    correction=True,
)


# --- Format ANOVA Results ---
# Cogntive Task
cogtsk_anova_overview = pd.concat(
    [
        format_rm_anova(gng_acc_aov, "Go–NoGo", "Accuracy"),
        format_rm_anova(gng_rct_aov, "Go–NoGo", "Reaction Time"),
        format_rm_anova(ds_ds_aov, "Digit Span", "Digit Span"),
        format_rm_anova(vs_acc_aov, "Visual Search", "Accuracy"),
        format_rm_anova(vs_rct_aov, "Visual Search", "Reaction Time"),
    ],
    ignore_index=True,
)


cogtsk_anova_overview.to_csv("cognitive_task_anova_overview.csv", index=False)

cogtsk_anova_overview.to_excel(
    "cognitive_task_anova_overview.xlsx", index=False, sheet_name="RM-ANOVA Overview"
)

# Mental Health
mh_anova_overview = pd.concat(
    [
        format_rm_anova(gad7_aov, "GAD7", "Total Score"),
        format_rm_anova(phq9_aov, "PHQ9", "Total Score"),
        format_rm_anova(wb_aov, "WEMWBS", "Total Score"),
    ],
    ignore_index=True,
)


mh_anova_overview.to_csv("mental_health_anova_overview.csv", index=False)

mh_anova_overview.to_excel(
    "mental_health_anova_overview.xlsx", index=False, sheet_name="RM-ANOVA Overview"
)

# Physiological and Perceptual Measurements
phys_per_anova_overview = pd.concat(
    [
        format_rm_anova(core_t_aov, "Core Temperature", "Score"),
        format_rm_anova(hr_aov, "Heart Rate", "Score"),
        format_rm_anova(tsen_aov, "Thermal Sensation", "Score"),
        format_rm_anova(tcom_aov, "Thermal Comfort", "Score"),
        format_rm_anova(rpe_aov, "RPE", "Score"),
    ],
    ignore_index=True,
)

phys_per_anova_overview.to_csv("phys_per_anova_overview.csv", index=False)

phys_per_anova_overview.to_excel(
    "phys_per_anova_overview.xlsx", index=False, sheet_name="RM-ANOVA Overview"
)

# Hoope Scale, Urine Osmolality
hooper_urine_anova_overview = pd.concat(
    [
        format_rm_anova(hooper_sleep_aov, "Sleep", "Sleep"),
        format_rm_anova(hooper_fatigue_aov, "Fatigue", "Fatigue"),
        format_rm_anova(hooper_ms_aov, "Muscle Soreness", "Muscle Soreness"),
        format_rm_anova(hooper_stress_aov, "Stress", "Stress"),
        format_rm_anova(hooper_urine_aov, "Urine Osmolality", "Urine Osmolality"),
    ],
    ignore_index=True,
)

hooper_urine_anova_overview.to_csv("hooper_urine_anova_overview.csv", index=False)

hooper_urine_anova_overview.to_excel(
    "hooper_urine_anova_overview.xlsx", index=False, sheet_name="RM-ANOVA Overview"
)
