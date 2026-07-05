"""Abkowitz-Polynommodell: RHS-Aufbau aus einer YAML-Koeffizientendatei.

YAML-Format (flaches Dict, Prime-System / SNAME):
    Name: [wert, [variablen]]   ->  Beitrag = wert * var1' * var2' * ...
    wert = null                 ->  Koeffizient wird ignoriert
    leere Variablenliste []     ->  konstanter Term (Bias)
    [udot]/[vdot]/[rdot]        ->  Beschleunigungsterm (Massenmatrix, LHS)
Der erste Buchstabe des Namens entscheidet die Gleichung (X / Y / N).
"""

import numpy as np
import sympy as sp

ACC_NAMES = ("udot", "vdot", "rdot")

# Erweiterungspunkt: Name -> Funktion, die den Prime-Wert der Variable aus
# Zustand y = [x0, y0, psi, u, v, r], Input inp = [N0, N1, delta_r0, delta_r1],
# Geschwindigkeit U und ShipConfig sd berechnet.
# Neue Variable (z.B. Propellerdrehzahl n) = eine Zeile hier ergaenzen.
VARIABLES = {
    "u":     lambda y, inp, U, sd: y[3] / U,
    "v":     lambda y, inp, U, sd: y[4] / U,
    "r":     lambda y, inp, U, sd: y[5] * sd.L / U,
    "delta": lambda y, inp, U, sd: 0.5 * (inp[2] + inp[3]),
    # "n":   lambda y, inp, U, sd: 0.5 * (inp[0] + inp[1]) * sd.L / U,
}


def build_abkowitz_equations(coeffs: dict, m_prime: float, iz_prime: float):
    """Baut die Bewegungsgleichungen im Prime-System auf und loest sie
    symbolisch nach den Beschleunigungen [udot', vdot', rdot'] auf.

    Returns:
        accs:      sp.Matrix([udot', vdot', rdot']) als Ausdruck in den Variablen
        forces:    sp.Matrix([X', Y', N']) nur Kraftterme (ohne Beschleunigungsterme)
        var_syms:  Liste der sympy-Symbole in der Reihenfolge von VARIABLES
    """
    var_syms = {name: sp.Symbol(name) for name in VARIABLES}
    acc_syms = {name: sp.Symbol(name) for name in ACC_NAMES}

    force = {"X": sp.Integer(0), "Y": sp.Integer(0), "N": sp.Integer(0)}
    acc_terms = {"X": sp.Integer(0), "Y": sp.Integer(0), "N": sp.Integer(0)}

    n_used = 0
    for name, (value, mult) in coeffs.items():
        if value is None or value == 0:
            continue
        eq = name[0].upper()
        if eq not in force:
            raise ValueError(f"Koeffizient '{name}': Name muss mit X, Y oder N beginnen")

        unknown = [m for m in mult if m not in VARIABLES and m not in ACC_NAMES]
        if unknown:
            raise ValueError(
                f"Koeffizient '{name}': unbekannte Variable(n) {unknown}, "
                f"erlaubt sind {list(VARIABLES) + list(ACC_NAMES)}"
            )

        if any(m in ACC_NAMES for m in mult):
            if len(mult) != 1:
                raise ValueError(
                    f"Koeffizient '{name}': Beschleunigungsterme muessen genau "
                    f"einen Multiplikator haben, bekommen: {mult}"
                )
            acc_terms[eq] += value * acc_syms[mult[0]]
        else:
            force[eq] += value * sp.Mul(*[var_syms[m] for m in mult])
        n_used += 1

    if n_used == 0:
        raise ValueError("Keine gesetzten Koeffizienten in der YAML (alle Werte null/0)")

    u, v, r = var_syms["u"], var_syms["v"], var_syms["r"]
    udot, vdot, rdot = (acc_syms[n] for n in ACC_NAMES)

    # Bewegungsgleichungen im Prime-System (Starrkoerperterme, xg vernachlaessigt)
    eqs = [
        sp.Eq(m_prime * (udot - v * r), force["X"] + acc_terms["X"]),
        sp.Eq(m_prime * (vdot + u * r), force["Y"] + acc_terms["Y"]),
        sp.Eq(iz_prime * rdot, force["N"] + acc_terms["N"]),
    ]
    sol = sp.solve(eqs, [udot, vdot, rdot], dict=True)
    if not sol:
        raise ValueError("Gleichungssystem nicht loesbar (Massenmatrix singulaer?)")
    sol = sol[0]

    accs = sp.Matrix([sol[udot], sol[vdot], sol[rdot]])
    forces = sp.Matrix([force["X"], force["Y"], force["N"]])
    return accs, forces, list(var_syms.values())


def make_abkowitz_rhs(coeffs: dict, sd, izz: float, input_data: np.ndarray,
                      t_input: np.ndarray, eps: float):
    """Baut eine rhs(t, y)-Funktion fuer solve_ivp aus den Abkowitz-Koeffizienten.

    Zustand y = [x0, y0, psi, u, v, r]; Inputs werden per Zero-Order-Hold aus
    input_data (Zeilen zu t_input) gehalten.

    Returns:
        rhs(t, y), forces_dim(t, y) -> (X, Y, N) in [N] bzw. [Nm]
    """
    m_prime = sd.m / (0.5 * sd.rho * sd.L**3)
    iz_prime = izz / (0.5 * sd.rho * sd.L**5)

    accs_sym, forces_sym, var_syms = build_abkowitz_equations(coeffs, m_prime, iz_prime)
    acc_func = sp.lambdify(var_syms, accs_sym, modules="numpy")
    force_func = sp.lambdify(var_syms, forces_sym, modules="numpy")
    prime_funcs = list(VARIABLES.values())
    L = sd.L

    def _primes(t, y):
        i = np.searchsorted(t_input, t, side="right") - 1
        inp = input_data[max(i, 0)]
        U = max(np.sqrt(y[3]**2 + y[4]**2), eps)
        return [f(y, inp, U, sd) for f in prime_funcs], U

    def rhs(t, y):
        vals, U = _primes(t, y)
        udot_p, vdot_p, rdot_p = np.asarray(acc_func(*vals)).ravel()
        psi = y[2]
        return np.array([
            y[3] * np.cos(psi) - y[4] * np.sin(psi),
            y[3] * np.sin(psi) + y[4] * np.cos(psi),
            y[5],
            udot_p * U**2 / L,
            vdot_p * U**2 / L,
            rdot_p * U**2 / L**2,
        ])

    def forces_dim(t, y):
        vals, U = _primes(t, y)
        Xp, Yp, Np = np.asarray(force_func(*vals)).ravel()
        q = 0.5 * sd.rho * L**2 * U**2
        return Xp * q, Yp * q, Np * q * L

    return rhs, forces_dim


if __name__ == "__main__":
    # Smoke-Test: null-Eintraege werden uebersprungen, [u, u] -> u**2
    test = {
        "Xudot": [-0.05, ["udot"]],
        "Xuu":   [-0.01, ["u", "u"]],
        "Xvv":   [None, ["v", "v"]],   # muss ignoriert werden
        "Yv":    [-0.2, ["v"]],
        "Yvdot": [-0.1, ["vdot"]],
        "Nr":    [-0.05, ["r"]],
        "Nd":    [-0.1, ["delta"]],
    }
    accs, forces, syms = build_abkowitz_equations(test, m_prime=0.01, iz_prime=0.001)
    sp.pprint(forces)
    sp.pprint(accs)
