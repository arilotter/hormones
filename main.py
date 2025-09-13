import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from models import MedicationType, Dosage, predict_hormone_curve

# e2 in pmol/L
# t in nmol/L

first_injection_date = "2025-04-17 12:00:00"
first_injection_dt = datetime.strptime(first_injection_date, "%Y-%m-%d %H:%M:%S")
hormone_data = [
    {
        "date": str(first_injection_dt - timedelta(weeks=2)),
        "estradiol": None,
        "testosterone": None,
        "dosage": 0,
        "notes": "e2 pills"
    },
    #   {
    #     "date": "2024-02-09 11:36:00",
    #     "estradiol": 50,
    #     "testosterone": 15.4,
    #     "dosage": None,
    #     "notes": "pre-hrt"
    # },
    #   {
    #     "date": "2024-07-03 14:47:00",
    #     "estradiol": 454,
    #     "testosterone": 7.3,
    #     "dosage": None,
    #     "notes": None
    # },
    # {
    #     "date": "2025-03-28 10:46:00",
    #     "estradiol": 6450,
    #     "testosterone": 2.6,
    #     "dosage": None,
    #     "notes": "silly outlier"
    # },
        {
        "date": "2025-04-03 9:31:00",
        "estradiol": 396,
        "testosterone": 2.1,
        "dosage": None,
        "notes": None
    },
    {
        "date": first_injection_date,
        "estradiol": None,
        "testosterone": None,
        "dosage": Dosage(MedicationType.ESTRADIOL_VALERATE, 6),
        "notes": "first injection",
    },
    {
        "date": "2025-05-05 11:07:00",
        "estradiol": 133,
        "testosterone": 1.5,
        "notes": None,
    },
    {
        "date": "2025-05-15 17:27:00",
        "estradiol": 46,
        "testosterone": 12.8,
        "notes": None,
    },
    {
        "date": "2025-05-26 10:30:00",
        "estradiol": 228,
        "testosterone": 6.5,
        "notes": None,
    },
    {
        "date": "2025-05-29 12:00:00",
        "estradiol": None,
        "testosterone": None,
        "dosage": Dosage(MedicationType.ESTRADIOL_VALERATE, 8),
        "notes": "20mg/ml vials",
    },
    {
        "date": "2025-06-02 11:53:00",
        "estradiol": 337,
        "testosterone": 4.4,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-06-05 11:17:00",
        "estradiol": 117,
        "testosterone": 16,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-06-07 14:48:00",
        "estradiol": 1522,
        "testosterone": 2.1,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-06-12 12:00:00",
        "estradiol": None,
        "testosterone": None,
        "dosage": None,
        "notes": "switched pharmacy, 10mg/ml vials",
    },
    {
        "date": "2025-06-12 12:58:00",
        "estradiol": 122,
        "testosterone": 19.6,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-06-14 14:39:00",
        "estradiol": 1997,
        "testosterone": 2.2,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-06-16 15:25:00",
        "estradiol": 1008,
        "testosterone": 3,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-07-03 12:00:00",
        "estradiol": None,
        "testosterone": None,
        "dosage": Dosage(MedicationType.ESTRADIOL_VALERATE, 10),
        "notes": "increased to 10mg injection",
    },
    {
        "date": "2025-07-10 12:08:00",
        "estradiol": 40,
        "testosterone": 14.7,
        "dosage": Dosage(MedicationType.ESTRADIOL_VALERATE, 10),
        "notes": None,
    },
    {
        "date": "2025-07-14 11:59:59",
        "estradiol": None,
        "testosterone": None,
        "dosage": None,
        "notes": "quit cigarettes",
    },
    {
        "date": "2025-07-17  14:14:00",
        "estradiol": 837,
        "testosterone": 1.9,
        "dosage": None,
        "notes": None,
    },
    {
        "date": "2025-07-24 14:07:00",
        "estradiol": 1294,
        "testosterone": 2.2,
        "dosage": None,
        "notes": None
    },
    # Example of how to add a different medication type:
    # {
    #     "date": "2025-08-01 12:00:00",
    #     "estradiol": None,
    #     "testosterone": None,
    #     "dosage": Dosage(MedicationType.DUMMY, 5),
    #     "notes": "switched to dummy medication for testing"
    # }
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
    """Prepare dosage data for step plotting by medication type"""
    # Sort by date
    df_sorted = df.sort_values("date").copy()

    # Forward fill dosage values to create step effect
    df_sorted["dosage_filled"] = df_sorted["dosage"].ffill()

    # Group by medication type
    dosage_by_type = {}
    
    current_dosage = None
    current_type = None

    for _, row in df_sorted.iterrows():
        if pd.notna(row["dosage_filled"]):
            dosage_obj = row["dosage_filled"]
            dosage_amount = dosage_obj.amount_mg if isinstance(dosage_obj, Dosage) else dosage_obj
            med_type = dosage_obj.medication_type if isinstance(dosage_obj, Dosage) else MedicationType.ESTRADIOL_VALERATE
            
            # Check if dosage or type changed
            if current_dosage != dosage_amount or current_type != med_type:
                if med_type not in dosage_by_type:
                    dosage_by_type[med_type] = {"dates": [], "values": []}
                
                dosage_by_type[med_type]["dates"].append(row["date"])
                dosage_by_type[med_type]["values"].append(dosage_amount)
                current_dosage = dosage_amount
                current_type = med_type

    return dosage_by_type


def generate_injection_schedule(start_dt, end_dt):
    """Generate injection schedule dates"""
    schedule = {
        "trough": [],  # Injection days (day 0)
        "peak": [],  # 2 days after injection
        "mid": [],  # 4 days after injection
    }

    current_injection_day = start_dt
    while current_injection_day <= end_dt:
        schedule["trough"].append(current_injection_day)
        schedule["peak"].append(current_injection_day + timedelta(days=2))
        schedule["mid"].append(current_injection_day + timedelta(days=4))
        current_injection_day += timedelta(weeks=1)

    return schedule


def categorize_bloodwork_by_cycle(test_dt, start_date=first_injection_date):
    """Categorize a bloodwork date as peak, mid, or trough based on weekly injection cycle"""
    start_dt = datetime.strptime(start_date[:10], "%Y-%m-%d")

    # Calculate days since start
    days_since_start = (test_dt - start_dt).days

    # before injections? who knows.
    if days_since_start < 0:
        return "?"

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



# from https://github.com/WHSAH/estrannaise.js
def generate_ev_expected_curve(df):
    """Generate expected estradiol valerate response curve using step dosage function"""
    # Generate injection schedule (every 7 days from start)
    start_date = pd.to_datetime(first_injection_date)
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
            return Dosage(MedicationType.ESTRADIOL_VALERATE, 6)  # Default starting dosage

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

                e2_contribution = predict_hormone_curve(days_since_injection, dose)
                total_e2 += e2_contribution

        expected_values.append(total_e2)

    return date_range.tolist(), expected_values


def generate_scaled_weekly_curves(df):
    """Generate weekly curves scaled to match actual data points"""
    start_date = pd.to_datetime(first_injection_date)

    # Get actual data points with hormone values
    df_with_data = df.dropna(subset=["estradiol", "testosterone"]).copy()
    df_with_data["cycle_category"] = df_with_data["date"].apply(
        categorize_bloodwork_by_cycle
    )

    scaled_curves = []

    # Find all unique weeks that contain data
    weeks_with_data = set()
    for _, row in df_with_data.iterrows():
        test_date = row["date"]
        days_since_start = (test_date - start_date).days
        week_number = days_since_start // 7
        weeks_with_data.add(week_number)

        # If this is a trough measurement, also add it to the previous week
        if row["cycle_category"] == "trough" and week_number > 0:
            weeks_with_data.add(week_number - 1)

    # Process each week that has data
    for week_number in sorted(weeks_with_data):
        injection_date = start_date + pd.Timedelta(weeks=week_number)

        # Get dosage for this injection
        dose = Dosage(MedicationType.ESTRADIOL_VALERATE, 6)  # Default
        dosage_changes = df[df["dosage"].notna() & (df["date"] <= injection_date)]
        if not dosage_changes.empty:
            dose = dosage_changes.iloc[-1]["dosage"]

        # Generate theoretical curve for this week (7 days)
        week_times = []
        week_values = []

        for hour in range(0, 7 * 24, 6):  # Every 6 hours for 7 days
            days = hour / 24.0
            week_times.append(injection_date + pd.Timedelta(days=days))
            week_values.append(predict_hormone_curve(days, dose))

        # Find actual data points in this week
        week_start = injection_date
        week_end = injection_date + pd.Timedelta(days=7)
        week_data = df_with_data[
            (df_with_data["date"] >= week_start) & (df_with_data["date"] < week_end)
        ]

        # Also include trough measurements from the next week (they represent end of this cycle)
        next_week_start = injection_date + pd.Timedelta(days=7)
        next_week_trough = df_with_data[
            (df_with_data["date"] >= next_week_start)
            & (df_with_data["date"] < next_week_start + pd.Timedelta(days=1))
            & (df_with_data["cycle_category"] == "trough")
        ]

        # Combine current week data with next week's trough
        all_week_data = pd.concat([week_data, next_week_trough], ignore_index=True)

        if not all_week_data.empty:
            # Separate measurements by type for better curve fitting
            peak_measurements = []
            end_trough_measurements = []
            other_measurements = []
            trough_baseline = 0

            for _, data_row in all_week_data.iterrows():
                # For trough measurements from next week, treat them as day 7 of current week
                if (
                    data_row["cycle_category"] == "trough"
                    and data_row["date"] >= next_week_start
                ):
                    days_since_injection = 7.0
                    end_trough_measurements.append(
                        (days_since_injection, data_row["estradiol"])
                    )
                else:
                    days_since_injection = (
                        data_row["date"] - injection_date
                    ).total_seconds() / (24 * 3600)

                    if days_since_injection < 0.1:  # Injection day trough
                        trough_baseline = data_row["estradiol"]
                    elif data_row["cycle_category"] == "peak":
                        peak_measurements.append(
                            (days_since_injection, data_row["estradiol"])
                        )
                    else:
                        other_measurements.append(
                            (days_since_injection, data_row["estradiol"])
                        )

            # If we have both peak and end trough, try to fit with time scaling
            if peak_measurements and end_trough_measurements:
                peak_day, peak_actual = peak_measurements[0]  # Take first peak
                end_day, end_trough_actual = end_trough_measurements[
                    0
                ]  # Take first end trough

                # Find the optimal time scaling factor
                best_time_scale = 1.0
                best_fit_error = float("inf")

                # Try different time scaling factors
                for time_scale in [x / 10.0 for x in range(5, 30)]:  # 0.5 to 3.0
                    # Scale the theoretical curve timing
                    scaled_peak_day = peak_day / time_scale
                    scaled_end_day = end_day / time_scale

                    # Get theoretical values at scaled times
                    theoretical_peak = predict_hormone_curve(scaled_peak_day, dose)
                    theoretical_end = predict_hormone_curve(scaled_end_day, dose)

                    if theoretical_peak > 0 and theoretical_end > 0:
                        # Calculate vertical scaling needed to match peak
                        vertical_scale = (
                            peak_actual - trough_baseline
                        ) / theoretical_peak

                        # Predict what the end trough should be with this scaling
                        predicted_end_trough = (
                            theoretical_end * vertical_scale + trough_baseline
                        )

                        # Calculate error between predicted and actual end trough
                        error = abs(predicted_end_trough - end_trough_actual)

                        if error < best_fit_error:
                            best_fit_error = error
                            best_time_scale = time_scale

                # Generate curve with optimal time scaling
                week_times = []
                week_values = []

                # Calculate vertical scaling based on peak with optimal time scaling
                scaled_peak_day = peak_day / best_time_scale
                theoretical_peak = predict_hormone_curve(scaled_peak_day, dose)
                vertical_scale = (
                    (peak_actual - trough_baseline) / theoretical_peak
                    if theoretical_peak > 0
                    else 1.0
                )

                for hour in range(0, 7 * 24, 6):  # Every 6 hours for 7 days
                    days = hour / 24.0
                    scaled_days = days / best_time_scale  # Apply time scaling
                    week_times.append(injection_date + pd.Timedelta(days=days))
                    theoretical_val = predict_hormone_curve(scaled_days, dose)
                    scaled_val = theoretical_val * vertical_scale + trough_baseline
                    week_values.append(scaled_val)

                scaled_curves.append(
                    {
                        "times": week_times,
                        "values": week_values,
                        "injection_date": injection_date,
                        "vertical_scaling": vertical_scale,
                        "time_scaling": best_time_scale,
                        "baseline_offset": trough_baseline,
                        "actual_points": all_week_data,
                        "week_number": week_number,
                        "fit_error": best_fit_error,
                    }
                )

            else:
                # Fallback to original method if we don't have both peak and end trough
                actual_values = []
                predicted_values = []

                for _, data_row in all_week_data.iterrows():
                    if (
                        data_row["cycle_category"] == "trough"
                        and data_row["date"] >= next_week_start
                    ):
                        days_since_injection = 7.0
                    else:
                        days_since_injection = (
                            data_row["date"] - injection_date
                        ).total_seconds() / (24 * 3600)

                    if days_since_injection < 0.1:  # Skip injection day
                        continue

                    predicted_val = predict_hormone_curve(days_since_injection, dose)
                    actual_values.append(data_row["estradiol"])
                    predicted_values.append(predicted_val)

                if predicted_values and all(p > 0 for p in predicted_values):
                    scaling_factors = [
                        a / p for a, p in zip(actual_values, predicted_values)
                    ]
                    avg_scaling = sum(scaling_factors) / len(scaling_factors)

                    # Generate standard scaled curve
                    week_times = []
                    week_values = []

                    for hour in range(0, 7 * 24, 6):
                        days = hour / 24.0
                        week_times.append(injection_date + pd.Timedelta(days=days))
                        week_values.append(
                            predict_hormone_curve(days, dose) * avg_scaling + trough_baseline
                        )

                    scaled_curves.append(
                        {
                            "times": week_times,
                            "values": week_values,
                            "injection_date": injection_date,
                            "vertical_scaling": avg_scaling,
                            "time_scaling": 1.0,
                            "baseline_offset": trough_baseline,
                            "actual_points": all_week_data,
                            "week_number": week_number,
                        }
                    )

    return scaled_curves


def create_hormone_graph(df):
    # Filter out rows with no hormone data for plotting
    df_with_data = df.dropna(subset=["estradiol", "testosterone"])

    # Add cycle categorization for bloodwork results
    df_with_data = df_with_data.copy()
    df_with_data["cycle_category"] = df_with_data["date"].apply(
        categorize_bloodwork_by_cycle
    )

    # Prepare dosage data
    dosage_by_type = prepare_dosage_data(df)
    
    # Define colors for medication types
    medication_colors = {
        MedicationType.ESTRADIOL_VALERATE: "green",
        MedicationType.DUMMY: "purple"
    }

    # Generate injection schedule
    injection_schedule = generate_injection_schedule(first_injection_dt, df.iloc[-1]["date"])

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
    cycle_colors = {"trough": "darkred", "peak": "green", "mid": "orange", "?": "blue"}
    cycle_markers = {"trough": "v", "peak": "^", "mid": "o", "?": "o"}

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

    # # Plot scaled weekly curves
    # for curve in scaled_curves:
    #     ax1.plot(
    #         curve["times"],
    #         curve["values"],
    #         "--",
    #         color="blue",
    #         linewidth=2,
    #         label="Scaled weekly curve" if curve == scaled_curves[0] else "",
    #     )

    # Plot points by cycle category
    for cycle_cat in ["trough", "peak", "mid", "?"]:
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
                zorder=5,
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
                zorder=5,
            )

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
    ax1.legend(loc='upper right')
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
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot Dosage as step plot by medication type
    if dosage_by_type:
        for med_type, data in dosage_by_type.items():
            dates = data["dates"]
            values = data["values"]
            color = medication_colors.get(med_type, "gray")
            med_name = med_type.value.replace("_", " ").title()
            
            ax3.step(
                dates,
                values,
                where="post",
                color=color,
                linewidth=3,
                marker="o",
                markersize=8,
                label=f"{med_name}",
            )

            # Fill the step areas for better visualization
            for i in range(len(dates)):
                if i < len(dates) - 1:
                    # Fill from current date to next date
                    ax3.fill_between(
                        [dates[i], dates[i + 1]],
                        values[i],
                        alpha=0.3,
                        color=color,
                        step="post",
                    )
                else:
                    # Fill from last date to end of plot
                    end_date = max(df["date"])
                    ax3.fill_between(
                        [dates[i], end_date],
                        values[i],
                        alpha=0.3,
                        color=color,
                        step="post",
                    )

    ax3.set_ylabel("Dosage (mg)", fontweight="bold")
    ax3.set_title("Medication Dosage by Type")
    ax3.legend(loc='upper right')
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
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)  # Start y-axis at 0

    # Add injection schedule markers to all plots
    for ax in [ax1, ax2, ax3, ax4]:
        for kind, dates in injection_schedule.items():
            for date in dates:
                ax.axvline(
                    x=date,
                    color=cycle_colors[kind],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )

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
            xy=(note_date, (ax1.get_ylim()[1])),
            xytext=(10, -30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            fontsize=8,
            rotation=0,
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
    xmin = min([a.get_xlim()[0] for a in (ax1, ax2, ax3, ax4)])
    xmax = max([a.get_xlim()[1] for a in (ax1, ax2, ax3, ax4)])
    for a in (ax1, ax2, ax3, ax4):
        a.set_xlim((xmin, xmax))
    return fig, (ax1, ax2, ax3, ax4)


# Create and display the graph
if __name__ == "__main__":
    # Print cycle categorization for all bloodwork results
    print("Bloodwork Results by Injection Cycle Position:")
    print("=" * 50)

    converted_data = convert_hormone_data(hormone_data)
    df = pd.DataFrame(converted_data)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    expected_curve_dates, expected_curve_values = generate_ev_expected_curve(df)

    for _, row in df.iterrows():
        cycle_cat = categorize_bloodwork_by_cycle(row["date"]) or "?"

        # Calculate days since start for reference
        start_dt = datetime.strptime("2025-04-17", "%Y-%m-%d")
        test_dt = row["date"]
        days_since_start = (test_dt - start_dt).days
        cycle_day = days_since_start % 7

        date_str = row["date"].strftime("%Y-%m-%d")
        print(f"{date_str}: {cycle_cat.upper()} (cycle day {cycle_day})")
        if row['testosterone'] == row['testosterone'] and row['estradiol'] == row['estradiol']:
            target_date = row["date"]
            closest_idx = min(
                range(len(expected_curve_dates)),
                key=lambda i: abs(
                    (expected_curve_dates[i] - target_date).total_seconds()
                ),
            )
            expected_value = expected_curve_values[closest_idx]
            ratio = row["estradiol"] / expected_value if expected_value > 0 else 0
            print(f"  E2: {row['estradiol']:.0f} pg/mL ({round(ratio  * 100)}% of predicted {expected_value:.0f} pg/mL), T: {row['testosterone']:.1f} ng/dL")
        if pd.notna(row['dosage']):
            print(f"  Dosage change: {row['dosage']}")
        if row['notes']:
            print(f"  Note: {row["notes"]}")
        print()

    fig, axes = create_hormone_graph(df)
    plt.savefig("hormone_levels.png", dpi=300, bbox_inches="tight")
    print("graph generated!")
