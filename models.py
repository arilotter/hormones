import numpy as np
from enum import Enum
from typing import Dict, Any


class MedicationType(Enum):
    ESTRADIOL_VALERATE = "estradiol_valerate"
    DUMMY = "dummy"


class Dosage:
    def __init__(self, medication_type: MedicationType, amount_mg: float):
        self.medication_type = medication_type
        self.amount_mg = amount_mg
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "medication_type": self.medication_type.value,
            "amount_mg": self.amount_mg
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dosage':
        return cls(
            medication_type=MedicationType(data["medication_type"]),
            amount_mg=data["amount_mg"]
        )
    
    def __str__(self) -> str:
        return f"{self.amount_mg}mg {self.medication_type.value}"


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
    elif dosage.medication_type == MedicationType.DUMMY:
        return dummy_model(t, dosage.amount_mg)
    else:
        raise ValueError(f"Unknown medication type: {dosage.medication_type}")