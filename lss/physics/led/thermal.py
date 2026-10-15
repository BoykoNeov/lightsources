"""LED Level 3: the drive heats the die and the phosphors (design §4.2 L3).

The heat the die cannot turn into light raises the junction above the board it
sits on; the junction temperature then moves the blue die's emission and the
phosphors' bands and quantum yields. This first step:

* **Junction temperature** T_j = T_board + R_th · (I·V − P_opt): the electrical
  input less the light that leaves, through one effective thermal resistance. The
  current, voltage and radiant power are the operating point's measured values, as
  incandescent Level 4's first step took the voltage; a model of the efficiency
  would predict them.
* **Phosphor temperature** equals the junction temperature: an assumption. A
  phosphor in silicone also heats from its own Stokes loss and can run hotter than
  the die.
* **Relative to Level 2.** Level 2's die peak, die width and phosphor constants
  were fitted at its ``phosphor_temperature``; Level 3 shifts them from there by
  the change in temperature, so at that temperature it is Level 2 exactly. Level
  2's quantum yields are 1, so a yield here is the yield relative to that anchor.
* **Only what changes colour.** The output is a relative spectrum. A die that
  loses efficiency as it heats, or droops with current, dims every part of the
  spectrum alike and is left out.
"""

import numpy as np

#: Boltzmann constant in eV/K.
K_EV = 8.617333262e-5
#: The temperatures the quenching data compare (25 °C and 150 °C).
T_ROOM_K = 298.15
T_HOT_K = 423.15


def junction_temperature(current_a: float, voltage_v: float, radiant_power_w: float, board_k: float,
                         thermal_resistance_k_per_w: float) -> float:
    """T_board + R_th · (I·V − P_opt): the heat left in the package raises the junction."""
    heat = current_a * voltage_v - radiant_power_w
    if heat < 0:
        raise ValueError(f"radiant power {radiant_power_w:g} W exceeds the electrical input {current_a * voltage_v:g} W")
    if thermal_resistance_k_per_w < 0:
        raise ValueError("thermal resistance must be ≥ 0")
    return board_k + thermal_resistance_k_per_w * heat


def quantum_yield(temperature_k: float, retained_hot: float, activation_ev: float) -> float:
    """Quantum yield η(T) = 1 / (1 + C·exp(−ΔE/kT)) of a phosphor that quenches over one barrier.

    The data give only ``retained_hot`` = η(150 °C)/η(25 °C); with the barrier ΔE
    assumed, that fixes C. η is then absolute in the sense of this form, which is 1
    at 0 K.
    """
    if not 0.0 < retained_hot <= 1.0:
        raise ValueError("the retained fraction at 150 °C must be in (0, 1]")
    if activation_ev <= 0:
        raise ValueError("the activation energy must be > 0")
    a, b = np.exp(-activation_ev / (K_EV * T_ROOM_K)), np.exp(-activation_ev / (K_EV * T_HOT_K))
    if retained_hot * b <= a:
        raise ValueError(f"a barrier of {activation_ev:g} eV cannot lose {1 - retained_hot:.0%} between 25 and 150 °C")
    c = (1.0 - retained_hot) / (retained_hot * b - a)
    return float(1.0 / (1.0 + c * np.exp(-activation_ev / (K_EV * temperature_k))))


def relative_yield(temperature_k: float, anchor_k: float, retained_hot: float, activation_ev: float) -> float:
    """η(T)/η(T_anchor): the yield relative to Level 2's anchor, where it is 1."""
    if temperature_k < anchor_k - 1e-9:
        raise ValueError(f"Level 3 cannot run the phosphors at {temperature_k:.1f} K, below Level 2's anchor "
                         f"({anchor_k:.1f} K): their yield relative to the anchor would exceed 1")
    return quantum_yield(temperature_k, retained_hot, activation_ev) / quantum_yield(anchor_k, retained_hot,
                                                                                      activation_ev)
