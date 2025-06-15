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
        "notes": "20mg/ml vials",
    },
    {
        "date": "2025-06-02",
        "estradiol": 337,
        "testosterone": 4.4,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-06-05",
        "estradiol": 117,
        "testosterone": 16,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-06-07",
        "estradiol": 1522,
        "testosterone": 2.1,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-06-12",
        "estradiol": None,
        "testosterone": None,
        "dosage": None,
        "notes": "switched pharmacy, 10mg/ml vials",
    },  {
        "date": "2025-06-12",
        "estradiol": 122,
        "testosterone": 19.6,
        "dosage": None,
        "notes":  None
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


def categorize_bloodwork_by_cycle(date_str, start_date="2025-04-17"):
    """Categorize a bloodwork date as peak, mid, or trough based on weekly injection cycle"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    test_dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Calculate days since start
    days_since_start = (test_dt - start_dt).days
    
    # Find position in weekly cycle (0-6)
    cycle_day = days_since_start % 7
    
    # Categorize based on cycle day
    # Day 0: Trough (injection day)
    # Day 1-3: Peak period (closest to day 2)
    # Day 4-6: Mid period (closest to day 4, approaching next trough)
    
    if cycle_day == 0:
        return "trough"
    elif 1 <= cycle_day <= 3:
        return "peak"
    else:  # 4 <= cycle_day <= 6
        return "mid"


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
    date_range = pd.date_range(start=start_date, end=end_date, freq="6h")
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


def generate_scaled_weekly_curves(df):
    """Generate weekly curves scaled to match actual data points"""
    start_date = pd.to_datetime("2025-04-17")
    
    # Get actual data points with hormone values
    df_with_data = df.dropna(subset=["estradiol", "testosterone"]).copy()
    df_with_data["cycle_category"] = df_with_data["date"].dt.strftime("%Y-%m-%d").apply(categorize_bloodwork_by_cycle)
    
    scaled_curves = []
    
    # Find all unique weeks that contain data
    weeks_with_data = set()
    for _, row in df_with_data.iterrows():
        test_date = row["date"]
        days_since_start = (test_date - start_date).days
        week_number = days_since_start // 7
        weeks_with_data.add(week_number)
    
    # Process each week that has data
    for week_number in sorted(weeks_with_data):
        injection_date = start_date + pd.Timedelta(weeks=week_number)
        
        # Get dosage for this injection
        dose = 6  # Default
        dosage_changes = df[df["dosage"].notna() & (df["date"] <= injection_date)]
        if not dosage_changes.empty:
            dose = dosage_changes.iloc[-1]["dosage"]
        
        # Generate theoretical curve for this week (7 days)
        week_times = []
        week_values = []
        
        for hour in range(0, 7*24, 6):  # Every 6 hours for 7 days
            days = hour / 24.0
            week_times.append(injection_date + pd.Timedelta(days=days))
            week_values.append(ev_model_3c(days, dose))
        
        # Find actual data points in this week
        week_start = injection_date
        week_end = injection_date + pd.Timedelta(days=7)
        week_data = df_with_data[(df_with_data["date"] >= week_start) & (df_with_data["date"] < week_end)]
        
        if not week_data.empty:
            # Calculate scaling factor based on actual vs predicted values
            actual_values = []
            predicted_values = []
            
            for _, data_row in week_data.iterrows():
                days_since_injection = (data_row["date"] - injection_date).total_seconds() / (24 * 3600)
                predicted_val = ev_model_3c(days_since_injection, dose)
                
                actual_values.append(data_row["estradiol"])
                predicted_values.append(predicted_val)
            
            # Calculate scaling factor (average ratio of actual to predicted)
            if predicted_values and all(p > 0 for p in predicted_values):
                scaling_factors = [a/p for a, p in zip(actual_values, predicted_values)]
                avg_scaling = sum(scaling_factors) / len(scaling_factors)
                
                # Scale the weekly curve
                scaled_week_values = [v * avg_scaling for v in week_values]
                
                scaled_curves.append({
                    'times': week_times,
                    'values': scaled_week_values,
                    'injection_date': injection_date,
                    'scaling_factor': avg_scaling,
                    'actual_points': week_data,
                    'week_number': week_number
                })
    
    return scaled_curves


def create_hormone_graph():
    converted_data = convert_hormone_data(hormone_data)

    # Convert data to DataFrame
    df = pd.DataFrame(converted_data)
    df["date"] = pd.to_datetime(df["date"])

    # Filter out rows with no hormone data for plotting
    df_with_data = df.dropna(subset=["estradiol", "testosterone"])
    
    # Add cycle categorization for bloodwork results
    df_with_data = df_with_data.copy()
    df_with_data["cycle_category"] = df_with_data["date"].dt.strftime("%Y-%m-%d").apply(categorize_bloodwork_by_cycle)

    # Prepare dosage data
    dosage_dates, dosage_values = prepare_dosage_data(df)

    # Generate injection schedule
    injection_schedule = generate_injection_schedule("2025-04-17")

    expected_curve_dates, expected_curve_values = generate_ev_expected_curve(df)
    
    # Generate scaled weekly curves
    scaled_curves = generate_scaled_weekly_curves(df)

    # Calculate ratio of actual to expected estradiol
    actual_to_expected_ratios = []
    ratio_dates = []

    # Desired ranges (hardcoded - adjust these values as needed)
    desired_estradiol_range = (100, 200)  # pg/mL
    desired_testosterone_range = (10, 50)  # ng/dL
    cis_man_estradiol_range = (0, 43.3)  # pg/mL
    cis_man_testosterone_range = (219, 905)  # ng/dL

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(25, 15))
    fig.suptitle(
        "Ari's Hormone Levels and Dosage Over Time", fontsize=16, fontweight="bold"
    )

    # Define colors and markers for cycle categories
    cycle_colors = {"trough": "darkred", "peak": "green", "mid": "orange"}
    cycle_markers = {"trough": "v", "peak": "^", "mid": "o"}

    for _, row in df_with_data.iterrows():
        cycle_cat = row["cycle_category"]
        
        if pd.notna(row["estradiol"]):
            # Find expected value for this date
            target_date = row["date"]
            closest_idx = min(
                range(len(expected_curve_dates)),
                key=lambda i: abs(
                    (expected_curve_dates[i] - target_date).total_seconds()
                ),
            )
            expected_value = expected_curve_values[closest_idx]
            ratio = row["estradiol"] / expected_value if expected_value > 0 else 0

            annotation_text = (
                f"{row['estradiol']:.0f} ({cycle_cat})\nTheory: {expected_value:.0f}"
            )
            ax1.annotate(
                annotation_text,
                xy=(row["date"], row["estradiol"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

            actual_to_expected_ratios.append(ratio)
            ratio_dates.append(target_date)

        if pd.notna(row["testosterone"]):
            annotation_text = f"{row['testosterone']:.1f} ({cycle_cat})"
            ax2.annotate(
                annotation_text,
                xy=(row["date"], row["testosterone"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
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
    
    # Plot scaled weekly curves
    for curve in scaled_curves:
        ax1.plot(
            curve['times'],
            curve['values'],
            "--",
            color="blue",
            linewidth=2,
            label="Scaled weekly curve" if curve == scaled_curves[0] else ""
        )

    # Plot points by cycle category
    for cycle_cat in ["trough", "peak", "mid"]:
        cycle_data = df_with_data[df_with_data["cycle_category"] == cycle_cat]
        if not cycle_data.empty:
            # Sort by date for proper line connections
            cycle_data_sorted = cycle_data.sort_values("date")
            
            # Plot points and connecting lines for each category
            ax1.plot(
                cycle_data_sorted["date"],
                cycle_data_sorted["estradiol"],
                "-",
                color=cycle_colors[cycle_cat],
                marker=cycle_markers[cycle_cat],
                markersize=8,
                linewidth=2,
                label=f"Estradiol ({cycle_cat})",
                zorder=5
            )
            
            # Plot testosterone with separate trend lines
            ax2.plot(
                cycle_data_sorted["date"],
                cycle_data_sorted["testosterone"],
                "-",
                color=cycle_colors[cycle_cat],
                marker=cycle_markers[cycle_cat],
                markersize=8,
                linewidth=2,
                label=f"Testosterone ({cycle_cat})",
                zorder=5
            )

    # Remove the overall connecting line for testosterone (we now have category-specific lines)

    ax1.axhspan(
        desired_estradiol_range[0],
        desired_estradiol_range[1],
        alpha=0.2,
        color="blue",
        label="Desired Range",
    )
    ax1.axhspan(
        cis_man_estradiol_range[0],
        cis_man_estradiol_range[1],
        alpha=0.2,
        color="red",
        label="Cis Man Range",
    )
    ax1.set_ylabel("Estradiol (pg/mL)", fontweight="bold")
    ax1.set_title("Estradiol Levels by Injection Cycle Position")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Testosterone
    ax2.axhspan(
        cis_man_testosterone_range[0],
        cis_man_testosterone_range[1],
        alpha=0.2,
        color="red",
        label="Cis Man Range",
    )
    ax2.axhspan(
        desired_testosterone_range[0],
        desired_testosterone_range[1],
        alpha=0.2,
        color="blue",
        label="Desired Range",
    )
    ax2.set_ylabel("Testosterone (ng/dL)", fontweight="bold")
    ax2.set_title("Testosterone Levels by Injection Cycle Position")
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

    # Plot Actual/Expected Ratio
    if ratio_dates and actual_to_expected_ratios:
        ax4.plot(
            ratio_dates,
            actual_to_expected_ratios,
            "o-",
            color="purple",
            linewidth=2,
            markersize=6,
            label="Actual/Expected Ratio",
        )

        # Add horizontal line at ratio = 1.0 (perfect match)
        ax4.axhline(
            y=1.0, color="gray", linestyle="-", alpha=0.5, label="Perfect Match"
        )

        # Add shaded region for reasonable variation (e.g., ±20%)
        ax4.axhspan(0.8, 1.2, alpha=0.2, color="gray", label="±20% Range")

        # Add ratio values as text annotations
        for date, ratio in zip(ratio_dates, actual_to_expected_ratios):
            ax4.annotate(
                f"{ratio:.2f}",
                xy=(date, ratio),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    ax4.set_ylabel("Ratio", fontweight="bold")
    ax4.set_title("Actual vs Expected Estradiol Ratio")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)  # Start y-axis at 0

    # Add injection schedule markers to all plots
    for ax in [ax1, ax2, ax3, ax4]:
        for kind, dates in injection_schedule.items():
            for date in dates:
                ax.axvline(x=date, color=cycle_colors[kind], linestyle="--", alpha=0.5, linewidth=1)

    # Add notes with vertical lines
    for _, row in df[df["notes"].notna()].iterrows():
        note_date = row["date"]
        note_text = row["notes"]

        # Add vertical line for notes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(
                x=note_date, color="black", linestyle="-", alpha=0.7, linewidth=2
            )

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
            [0], [0], color="darkred", linestyle="--", label="Trough (Injection Day)"
        ),
        plt.Line2D([0], [0], color="red", linestyle=":", label="Peak (Day +2)"),
        plt.Line2D([0], [0], color="orange", linestyle="-.", label="Mid (Day +4)"),
        plt.Line2D([0], [0], color="black", linestyle="-", alpha=0.7, label="Notes"),
    ]

    fig.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1.15, 0.5))

    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax4.set_xlabel("Date", fontweight="bold")

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    return fig, (ax1, ax2, ax3, ax4)


# Create and display the graph
if __name__ == "__main__":
    # Print cycle categorization for all bloodwork results
    print("Bloodwork Results by Injection Cycle Position:")
    print("=" * 50)
    
    converted_data = convert_hormone_data(hormone_data)
    df = pd.DataFrame(converted_data)
    df["date"] = pd.to_datetime(df["date"])
    df_with_data = df.dropna(subset=["estradiol", "testosterone"])
    
    for _, row in df_with_data.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        cycle_cat = categorize_bloodwork_by_cycle(date_str)
        
        # Calculate days since start for reference
        start_dt = datetime.strptime("2025-04-17", "%Y-%m-%d")
        test_dt = row["date"]
        days_since_start = (test_dt - start_dt).days
        cycle_day = days_since_start % 7
        
        print(f"{date_str}: {cycle_cat.upper()} (cycle day {cycle_day})")
        print(f"  E2: {row['estradiol']:.0f} pg/mL, T: {row['testosterone']:.1f} ng/dL")
        print()
    
    fig, axes = create_hormone_graph()
    plt.savefig("hormone_levels.png", dpi=300, bbox_inches="tight")
    print("graph generated!")
