import numpy as np
from enum import Enum
from typing import Dict, Any


class MedicationType(Enum):
    NO_HRT = "no_hrt"
    ORAL = "oral"
    ESTRADIOL_VALERATE = "estradiol_valerate"
    ESTRADIOL_ENANTHATE = "estradiol_enanthate"
    DUMMY = "dummy"


class Dosage:
    def __init__(self, medication_type: MedicationType, amount_mg: float, interval_days: float = 7.0):
        self.medication_type = medication_type
        self.amount_mg = amount_mg
        self.interval_days = interval_days
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "medication_type": self.medication_type.value,
            "amount_mg": self.amount_mg,
            "interval_days": self.interval_days
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dosage':
        return cls(
            medication_type=MedicationType(data["medication_type"]),
            amount_mg=data["amount_mg"],
            interval_days=data.get("interval_days", 7.0)
        )
    
    def __str__(self) -> str:
        return f"{self.amount_mg}mg {self.medication_type.value} every {self.interval_days} days"


def ev_model_3c(t, dose_mg):
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
        ret = dose_mg * d * k1 * k1 * t * t * np.exp(-k1 * t) / 2
    elif k1 == k2 and k2 != k3:
        ret = (
            dose_mg
            * d
            * k1
            * k1
            * (np.exp(-k3 * t) - np.exp(-k1 * t) * (1 + (k1 - k3) * t))
            / (k1 - k3)
            / (k1 - k3)
        )
    elif k1 != k2 and k1 == k3:
        ret = (
            dose_mg
            * d
            * k1
            * k2
            * (np.exp(-k2 * t) - np.exp(-k1 * t) * (1 + (k1 - k2) * t))
            / (k1 - k2)
            / (k1 - k2)
        )
    elif k1 != k2 and k2 == k3:
        ret = (
            dose_mg
            * d
            * k1
            * k2
            * (np.exp(-k1 * t) - np.exp(-k2 * t) * (1 - (k1 - k2) * t))
            / (k1 - k2)
            / (k1 - k2)
        )
    else:
        ret = (
            dose_mg
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


def een_model_3c(t, dose_mg):
    """
    Estradiol enanthate 3-compartment model using parameters from reference implementation
    Parameters: [191.4, 0.119, 0.601, 0.402]
    """
    if t < 0:
        return 0

    # EEn IM parameters from the reference model
    d = 191.4   # bioavailability factor
    k1 = 0.119  # absorption rate constant (1/day)
    k2 = 0.601  # distribution rate constant (1/day)
    k3 = 0.402  # elimination rate constant (1/day)

    # 3-compartment model calculation (same formula as EV, different parameters)
    if k1 == k2 and k2 == k3:
        ret = dose_mg * d * k1 * k1 * t * t * np.exp(-k1 * t) / 2
    elif k1 == k2 and k2 != k3:
        ret = (
            dose_mg
            * d
            * k1
            * k1
            * (np.exp(-k3 * t) - np.exp(-k1 * t) * (1 + (k1 - k3) * t))
            / (k1 - k3)
            / (k1 - k3)
        )
    elif k1 != k2 and k1 == k3:
        ret = (
            dose_mg
            * d
            * k1
            * k2
            * (np.exp(-k2 * t) - np.exp(-k1 * t) * (1 + (k1 - k2) * t))
            / (k1 - k2)
            / (k1 - k2)
        )
    elif k1 != k2 and k2 == k3:
        ret = (
            dose_mg
            * d
            * k1
            * k2
            * (np.exp(-k1 * t) - np.exp(-k2 * t) * (1 - (k1 - k2) * t))
            / (k1 - k2)
            / (k1 - k2)
        )
    else:
        ret = (
            dose_mg
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


def dummy_model(t, dose_mg):
    """
    Simple dummy model: y = x * dose_mg
    """
    if t < 0:
        return 0
    return t * dose_mg


def predict_hormone_curve(t, dosage: Dosage):
    """
    Predict hormone curve based on dosage type
    
    Args:
        t: time in days since injection
        dosage: Dosage object
    
    Returns:
        Predicted hormone level
    """
    if dosage.medication_type == MedicationType.ESTRADIOL_VALERATE:
        return ev_model_3c(t, dosage.amount_mg)
    elif dosage.medication_type == MedicationType.ESTRADIOL_ENANTHATE:
        return een_model_3c(t, dosage.amount_mg)
    elif dosage.medication_type == MedicationType.DUMMY:
        return dummy_model(t, dosage.amount_mg)
    elif dosage.medication_type == MedicationType.ORAL or dosage.medication_type == MedicationType.NO_HRT:
        return 0
    else:
        raise ValueError(f"Unknown medication type: {dosage.medication_type}")


def calculate_cycle_points_for_injection(injection_date, dosage: Dosage, all_previous_injections, samples_per_day: int = 24):
    """
    Calculate dynamic peak, mid, and trough points for a specific injection cycle,
    accounting for steady-state accumulation from all previous injections.
    
    Args:
        injection_date: Date of current injection
        dosage: Dosage object for current injection
        all_previous_injections: List of (date, dosage) tuples for all previous injections
        samples_per_day: Number of samples per day for curve evaluation
    
    Returns:
        dict: {
            'peak_date': datetime,    # Date with highest predicted value in this cycle
            'trough_date': datetime,  # Date of next injection (lowest predicted value)
            'mid_date': datetime      # Date with 50% value between peak and trough
        }
    """
    from datetime import timedelta
    
    interval = dosage.interval_days
    time_step = 1.0 / samples_per_day
    
    # Sample points during this injection cycle
    sample_dates = []
    predicted_values = []
    
    current_time = 0
    while current_time <= interval:
        sample_date = injection_date + timedelta(days=current_time)
        sample_dates.append(sample_date)
        
        # Calculate total hormone level from ALL injections (current + previous)
        total_level = 0
        
        # Contribution from current injection
        total_level += predict_hormone_curve(current_time, dosage)
        
        # Contributions from all previous injections
        for prev_date, prev_dosage in all_previous_injections:
            days_since_prev = (sample_date - prev_date).total_seconds() / (24 * 3600)
            if days_since_prev >= 0:  # Only count injections that already happened
                total_level += predict_hormone_curve(days_since_prev, prev_dosage)
        
        predicted_values.append(total_level)
        current_time += time_step
    
    predicted_values = np.array(predicted_values)
    
    # Find peak (maximum value in this cycle)
    peak_idx = np.argmax(predicted_values)
    peak_date = sample_dates[peak_idx]
    peak_value = predicted_values[peak_idx]
    
    # Trough is ALWAYS the next injection date (end of interval)
    trough_date = injection_date + timedelta(days=interval)
    trough_value = predicted_values[-1]  # Last sample point
    
    # Find mid point (50% between peak and trough values)
    target_mid_value = trough_value + 0.5 * (peak_value - trough_value)
    
    # Find ALL points close to the target mid value (within 5% tolerance)
    tolerance = 0.05 * (peak_value - trough_value)
    mid_candidates = []
    
    for i, value in enumerate(predicted_values):
        if abs(value - target_mid_value) <= tolerance:
            mid_candidates.append(i)
    
    if not mid_candidates:
        # Fallback: just use closest point
        differences = np.abs(predicted_values - target_mid_value)
        mid_idx = np.argmin(differences)
    else:
        # Pick the candidate that maximizes spacing from both peak and trough
        best_mid_idx = mid_candidates[0]
        best_spacing = 0
        
        for candidate_idx in mid_candidates:
            candidate_date = sample_dates[candidate_idx]
            
            # Calculate minimum distance to either peak or trough
            dist_to_peak = abs((candidate_date - peak_date).total_seconds())
            dist_to_trough = abs((candidate_date - trough_date).total_seconds())
            min_spacing = min(dist_to_peak, dist_to_trough)
            
            if min_spacing > best_spacing:
                best_spacing = min_spacing
                best_mid_idx = candidate_idx
        
        mid_idx = best_mid_idx
    
    mid_date = sample_dates[mid_idx]
    
    return {
        'peak_date': peak_date,
        'trough_date': trough_date, 
        'mid_date': mid_date,
        'peak_value': float(peak_value),
        'trough_value': float(trough_value),
        'mid_value': float(target_mid_value)
    }