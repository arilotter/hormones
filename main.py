import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# e2 in pmol/L
# t in nmol/L
# dosage in mg/injection

hormone_data = [
    {
        "date": "2025-04-17",
        "estradiol": None,
        "testosterone": None,
        "dosage": 6,
        "notes": "first injection",
    },
    {
        "date": "2025-05-05",
        "estradiol": 133,
        "testosterone": 1.5,
        "notes": None,
    },
    {
        "date": "2025-05-15",
        "estradiol": 46,
        "testosterone": 12.8,
        "notes": None,
    },
    {
        "date": "2025-05-26",
        "estradiol": 228,
        "testosterone": 6.5,
        "notes": None,
    },
    {
        "date": "2025-05-29",
        "estradiol": None,
        "testosterone": None,
        "dosage": 8,
        "notes": "dosage increase",
    },
    {
        "date": "2025-06-02",
        "estradiol": 337,
        "testosterone": 4.4,
        "dosage": None,
        "notes": None,
    },
]


def convert_hormone_data(data):
    """Convert hormone data to target units"""
    converted_data = []

    for row in data:
        converted_row = row.copy()

        # Convert estradiol from pmol/L to pg/mL
        converted_row["estradiol"] = (
            None if row["estradiol"] is None else row["estradiol"] * 0.2724
        )

        # Convert testosterone from nmol/L to ng/dL
        converted_row["testosterone"] = (
            None if row["testosterone"] is None else row["testosterone"] * 28.842
        )

        converted_data.append(converted_row)

    return converted_data


def prepare_dosage_data(df):
    """Prepare dosage data for step plotting"""
    # Sort by date
    df_sorted = df.sort_values("date").copy()

    # Forward fill dosage values to create step effect
    df_sorted["dosage_filled"] = df_sorted["dosage"].ffill()

    # Create step plot data
    dosage_dates = []
    dosage_values = []

    current_dosage = None

    for _, row in df_sorted.iterrows():
        if pd.notna(row["dosage_filled"]):
            if current_dosage != row["dosage_filled"]:
                # Add the new dosage level
                dosage_dates.append(row["date"])
                dosage_values.append(row["dosage_filled"])
                current_dosage = row["dosage_filled"]

    return dosage_dates, dosage_values


def generate_injection_schedule(start_date, num_weeks=12):
    """Generate injection schedule dates"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    schedule = {
        "trough": [],  # Injection days (day 0)
        "peak": [],  # 2 days after injection
        "mid": [],  # 4 days after injection
    }

    for week in range(num_weeks):
        injection_day = start_dt + timedelta(weeks=week)
        schedule["trough"].append(injection_day)
        schedule["peak"].append(injection_day + timedelta(days=2))
        schedule["mid"].append(injection_day + timedelta(days=4))

    return schedule


def ev_model_3c(t, dose):
    """
    Estradiol valerate 3-compartment model using parameters from JS model
    Parameters from PKParameters["EV im"]: [478.0, 0.236, 4.85, 1.24]
    """
    if t < 0:
        return 0

    # EV IM parameters from the JS model
    d = 478.0  # bioavailability factor
    k1 = 0.236  # absorption rate constant (1/day)
    k2 = 4.85  # distribution rate constant (1/day)
    k3 = 1.24  # elimination rate constant (1/day)

    # 3-compartment model calculation (matching JS e2Curve3C function)
    if k1 == k2 and k2 == k3:
        ret = dose * d * k1 * k1 * t * t * np.exp(-k1 * t) / 2
    elif k1 == k2 and k2 != k3:
        ret = (
            dose
            * d
            * k1
            * k1
            * (np.exp(-k3 * t) - np.exp(-k1 * t) * (1 + (k1 - k3) * t))
            / (k1 - k3)
            / (k1 - k3)
        )
    elif k1 != k2 and k1 == k3:
        ret = (
            dose
            * d
            * k1
            * k2
            * (np.exp(-k2 * t) - np.exp(-k1 * t) * (1 + (k1 - k2) * t))
            / (k1 - k2)
            / (k1 - k2)
        )
    elif k1 != k2 and k2 == k3:
        ret = (
            dose
            * d
            * k1
            * k2
            * (np.exp(-k1 * t) - np.exp(-k2 * t) * (1 - (k1 - k2) * t))
            / (k1 - k2)
            / (k1 - k2)
        )
    else:
        ret = (
            dose
            * d
            * k1
            * k2
            * (
                np.exp(-k1 * t) / ((k1 - k2) * (k1 - k3))
                - np.exp(-k2 * t) / ((k1 - k2) * (k2 - k3))
                + np.exp(-k3 * t) / ((k1 - k3) * (k2 - k3))
            )
        )

    return max(0, ret)


# from https://github.com/WHSAH/estrannaise.js
def generate_ev_expected_curve(df):
    """Generate expected estradiol valerate response curve using step dosage function"""
    # Generate injection schedule (every 7 days from start)
    start_date = pd.to_datetime("2025-04-17")
    end_date = df["date"].max() + pd.Timedelta(days=14)

    # Create weekly injection dates
    injection_dates = []
    current_date = start_date
    while current_date <= end_date:
        injection_dates.append(current_date)
        current_date += pd.Timedelta(days=7)

    # Get dosage step function data
    df_sorted = df.sort_values("date").copy()
    df_sorted["dosage_filled"] = df_sorted["dosage"].ffill()

    # Create dosage lookup function
    def get_dosage_at_date(target_date):
        # Find the most recent dosage change before or on target_date
        valid_dosages = df_sorted[
            (df_sorted["date"] <= target_date) & (df_sorted["dosage_filled"].notna())
        ]

        if valid_dosages.empty:
            return 6  # Default starting dosage

        return valid_dosages.iloc[-1]["dosage_filled"]

    # Generate time points for curve
    date_range = pd.date_range(start=start_date, end=end_date, freq="6H")
    expected_values = []

    for current_date in date_range:
        total_e2 = 0

        # Sum contributions from all weekly injections
        for injection_date in injection_dates:
            if injection_date <= current_date:  # Only past injections
                days_since_injection = (
                    current_date - injection_date
                ).total_seconds() / (24 * 3600)

                # Get the dosage that was active at this injection date
                dose = get_dosage_at_date(injection_date)

                e2_contribution = ev_model_3c(days_since_injection, dose)
                total_e2 += e2_contribution

        expected_values.append(total_e2)

    return date_range.tolist(), expected_values


def create_hormone_graph():
    converted_data = convert_hormone_data(hormone_data)

    # Convert data to DataFrame
    df = pd.DataFrame(converted_data)
    df["date"] = pd.to_datetime(df["date"])

    # Filter out rows with no hormone data for plotting
    df_with_data = df.dropna(subset=["estradiol", "testosterone"])

    # Prepare dosage data
    dosage_dates, dosage_values = prepare_dosage_data(df)

    # Generate injection schedule
    injection_schedule = generate_injection_schedule("2025-04-17")

    expected_curve_dates, expected_curve_values = generate_ev_expected_curve(df)

    # Desired ranges (hardcoded - adjust these values as needed)
    estradiol_range = (100, 200)  # pg/mL
    testosterone_range = (10, 50)  # ng/dL

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        "Ari's Hormone Levels and Dosage Over Time", fontsize=16, fontweight="bold"
    )

    # Plot expected estradiol curve
    if expected_curve_dates and expected_curve_values:
        ax1.plot(
            expected_curve_dates,
            expected_curve_values,
            "--",
            color="pink",
            linewidth=2,
            label="Expected E2 (EV model)",
        )

    ax1.plot(
        df_with_data["date"],
        df_with_data["estradiol"],
        "o-",
        color="red",
        linewidth=2,
        markersize=6,
        label="Estradiol",
    )
    ax1.axhspan(
        estradiol_range[0],
        estradiol_range[1],
        alpha=0.2,
        color="red",
        label="Desired Range",
    )
    ax1.set_ylabel("Estradiol (pg/mL)", fontweight="bold")
    ax1.set_title("Estradiol Levels")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Testosterone
    ax2.plot(
        df_with_data["date"],
        df_with_data["testosterone"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=6,
        label="Testosterone",
    )
    ax2.axhspan(
        testosterone_range[0],
        testosterone_range[1],
        alpha=0.2,
        color="blue",
        label="Desired Range",
    )
    ax2.set_ylabel("Testosterone (ng/dL)", fontweight="bold")
    ax2.set_title("Testosterone Levels")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot Dosage as step plot
    if dosage_dates and dosage_values:
        ax3.step(
            dosage_dates,
            dosage_values,
            where="post",
            color="green",
            linewidth=3,
            marker="o",
            markersize=8,
            label="Dosage",
        )

        # Fill the step areas for better visualization
        for i in range(len(dosage_dates)):
            if i < len(dosage_dates) - 1:
                # Fill from current date to next date
                ax3.fill_between(
                    [dosage_dates[i], dosage_dates[i + 1]],
                    dosage_values[i],
                    alpha=0.3,
                    color="green",
                    step="post",
                )
            else:
                # Fill from last date to end of plot
                end_date = max(df["date"])
                ax3.fill_between(
                    [dosage_dates[i], end_date],
                    dosage_values[i],
                    alpha=0.3,
                    color="green",
                    step="post",
                )

    ax3.set_ylabel("Dosage (mg)", fontweight="bold")
    ax3.set_title("Dosage Changes")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)  # Start y-axis at 0 for dosage

    # Add injection schedule markers to all plots
    for ax in [ax1, ax2, ax3]:
        # Trough markers (injection days)
        for date in injection_schedule["trough"]:
            ax.axvline(x=date, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        # Peak markers (2 days after injection)
        for date in injection_schedule["peak"]:
            ax.axvline(x=date, color="orange", linestyle=":", alpha=0.5, linewidth=1)

        # Mid markers (4 days after injection)
        for date in injection_schedule["mid"]:
            ax.axvline(x=date, color="purple", linestyle="-.", alpha=0.5, linewidth=1)

    # Add notes with vertical lines
    for _, row in df[df["notes"].notna()].iterrows():
        note_date = row["date"]
        note_text = row["notes"]

        # Add vertical line for notes
        for ax in [ax1, ax2, ax3]:
            ax.axvline(
                x=note_date, color="black", linestyle="-", alpha=0.7, linewidth=2
            )

        # Add note text to the top plot
        ax1.annotate(
            note_text,
            xy=(note_date, (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 2),
            xytext=(10, -30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            fontsize=8,
            rotation=45,
        )

    legend_elements = [
        plt.Line2D(
            [0], [0], color="gray", linestyle="--", label="Trough (Injection Day)"
        ),
        plt.Line2D([0], [0], color="orange", linestyle=":", label="Peak (Day +2)"),
        plt.Line2D([0], [0], color="purple", linestyle="-.", label="Mid (Day +4)"),
        plt.Line2D([0], [0], color="black", linestyle="-", alpha=0.7, label="Notes"),
    ]

    fig.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1.15, 0.5))

    # Format x-axis
    ax3.set_xlabel("Date", fontweight="bold")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    return fig, (ax1, ax2, ax3)


# Create and display the graph
if __name__ == "__main__":
    fig, axes = create_hormone_graph()
    plt.savefig("hormone_levels.png", dpi=300, bbox_inches="tight")
