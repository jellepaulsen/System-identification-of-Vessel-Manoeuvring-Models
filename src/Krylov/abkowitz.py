"""Abkowitz-Polynommodell: Kraefte und Bewegungsgleichungen aus einer YAML.

YAML-Format (flaches Dict, Prime-System, siehe SIMULATOR.md):
    Name: [wert, [variablen]]   ->  Beitrag = wert * var1' * var2' * ...
    wert = null                 ->  Koeffizient wird ignoriert
    leere Variablenliste []     ->  konstanter Term (Bias)
    [udot]/[vdot]/[rdot]        ->  Beschleunigungsterm (Massenmatrix, LHS)
Der erste Buchstabe des Namens entscheidet die Gleichung (X / Y / N).
Masse, stat. Moment und Traegheit (m', mxg', Iz') werden aus den Schiffsdaten
berechnet (mass_primes_from_ship, xg = lcg - L/2); alte m/mxg/Iz-Eintraege in
der YAML werden ignoriert.

Normierung (D = Tiefgang Tm, U = momentane Geschwindigkeit):
    m, mxg : 0.5*rho*L^2*D      Iz     : 0.5*rho*L^4*D
    X, Y   : 0.5*rho*L*D*U^2    N      : 0.5*rho*L^2*D*U^2
    u' = u/U, v' = v/U, r' = r*L/U;  udot = udot'*U^2/L, rdot = rdot'*U^2/L^2
"""

import numpy as np
import sympy as sp
import yaml

ACC_NAMES = ("udot", "vdot", "rdot")
MASS_KEYS = ("m", "mxg", "Iz")

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


def mass_primes_from_ship(sd):
    """Berechnet m', mxg', Iz' aus den Schiffsdaten (Prime-System mit D = Tm).

    xg = lcg - L/2 (Abstand des Schwerpunkts von Lpp/2, positiv nach vorn).
    Normierung: m', mxg'/L : 0.5*rho*L^2*D   |   Iz' : 0.5*rho*L^4*D
    """
    if sd.m is None or sd.I_z is None or sd.lcg is None:
        raise ValueError("mass_primes_from_ship braucht m, I_z und lcg in den Schiffsdaten")
    denom_m = 0.5 * sd.rho * sd.L**2 * sd.Tm
    xg = sd.lcg - sd.L / 2.0
    m_prime = sd.m / denom_m
    mxg_prime = sd.m * xg / (denom_m * sd.L)
    iz_prime = sd.I_z / (denom_m * sd.L**2)
    return m_prime, mxg_prime, iz_prime


def split_mass_entries(coeffs: dict):
    """Entfernt die frueheren Skalar-Schluessel m/mxg/Iz aus dem Koeffizienten-Dict.

    Die Massen kommen inzwischen aus den Schiffsdaten (mass_primes_from_ship);
    noch vorhandene YAML-Eintraege werden nur zurueckgegeben, damit der Aufrufer
    sie melden kann.
    """
    coeffs = dict(coeffs)
    ignored = {k: coeffs.pop(k) for k in MASS_KEYS if k in coeffs}
    return ignored, coeffs


def build_abkowitz_equations(coeffs: dict, m_prime: float, mxg_prime: float,
                             iz_prime: float):
    """Baut die Bewegungsgleichungen im Prime-System auf und loest sie
    symbolisch nach den Beschleunigungen [udot', vdot', rdot'] auf.

    Neben den Rumpfpolynomen wirken die externen Zusatzkraefte X_ext/Y_ext/N_ext
    (prime, z.B. Pod-/Ruderkraefte) auf der rechten Seite - modular wie bei
    Krylov oder MMG. Ohne externes Modul einfach 0 einsetzen.

    Returns:
        accs:      sp.Matrix([udot', vdot', rdot']) als Ausdruck in den Variablen
        forces:    sp.Matrix([X', Y', N']) nur Rumpf-Kraftterme (ohne acc/ext)
        var_syms:  Liste der sympy-Symbole in der Reihenfolge von VARIABLES
        ext_syms:  [X_ext, Y_ext, N_ext]
    """
    var_syms = {name: sp.Symbol(name) for name in VARIABLES}
    acc_syms = {name: sp.Symbol(name) for name in ACC_NAMES}

    force = {"X": sp.Integer(0), "Y": sp.Integer(0), "N": sp.Integer(0)}
    acc_terms = {"X": sp.Integer(0), "Y": sp.Integer(0), "N": sp.Integer(0)}

    n_used = 0
    for name, entry in coeffs.items():
        try:
            value, mult = entry
        except (TypeError, ValueError):
            raise ValueError(f"Koeffizient '{name}': erwartet [wert, [variablen]], "
                             f"bekommen: {entry!r}")
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
    ext_syms = [sp.Symbol("X_ext"), sp.Symbol("Y_ext"), sp.Symbol("N_ext")]

    # Bewegungsgleichungen im Prime-System inkl. xg-Kopplung
    eqs = [
        sp.Eq(m_prime * (udot - v * r) - mxg_prime * r**2,
              force["X"] + acc_terms["X"] + ext_syms[0]),
        sp.Eq(m_prime * (vdot + u * r) + mxg_prime * rdot,
              force["Y"] + acc_terms["Y"] + ext_syms[1]),
        sp.Eq(iz_prime * rdot + mxg_prime * (vdot + u * r),
              force["N"] + acc_terms["N"] + ext_syms[2]),
    ]
    sol = sp.solve(eqs, [udot, vdot, rdot], dict=True)
    if not sol:
        raise ValueError("Gleichungssystem nicht loesbar (Massenmatrix singulaer?)")
    sol = sol[0]

    accs = sp.Matrix([sol[udot], sol[vdot], sol[rdot]])
    forces = sp.Matrix([force["X"], force["Y"], force["N"]])
    return accs, forces, list(var_syms.values()), ext_syms


def pod_forces_from_krylov(kf):
    """Adapter: calc_pod_rud_force einer Krylov_forces-Instanz als Pod-Modul.

    Rueckgabe: pod(t, y, inp) -> (X, Y, N) dimensional [N, Nm].
    """
    def pod(t, y, inp):
        return kf.calc_pod_rud_force(input=inp, input_columns=kf.input_columns,
                                     sd=kf.sd, urx=y[3])
    return pod


class AbkowitzModel:
    """Modell fuer den Simulator (siehe SIMULATOR.md): Abkowitz-Polynom im
    Prime-System, Koeffizienten und Massen (m, mxg, Iz) aus der YAML.

    Optional modular wie Krylov/MMG: mit pod_forces (Funktion
    (t, y, inp) -> (X, Y, N) dimensional, z.B. pod_forces_from_krylov(kf))
    wirken Pod-/Ruderkraefte zusaetzlich zu den Rumpfpolynomen. Die Auswahl
    faellt beim Registrieren/Starten der Simulation ueber den Modellnamen,
    z.B. sim.add_model("abkowitz_pod", AbkowitzModel.from_yaml(..., pod_forces=...)).
    """

    def __init__(self, coeffs: dict, sd, eps: float, pod_forces=None):
        self.sd = sd
        self.eps = eps
        self.pod_forces = pod_forces
        self.force_columns = (["hull_X", "hull_Y", "hull_N",
                               "pod_X", "pod_Y", "pod_N",
                               "F_X", "F_Y", "M_N"]
                              if pod_forces is not None
                              else ["F_X", "F_Y", "M_N"])
        ignored, coeffs = split_mass_entries(coeffs)
        m_prime, mxg_prime, iz_prime = mass_primes_from_ship(sd)
        print(f"AbkowitzModel: Massen aus Schiffsdaten: m'={m_prime:.4f}, "
              f"mxg'={mxg_prime:.5f} (xg={sd.lcg - sd.L/2:+.2f} m), Iz'={iz_prime:.5f}")
        if ignored:
            print(f"AbkowitzModel: YAML-Eintraege {ignored} werden ignoriert "
                  f"(Massen kommen aus den Schiffsdaten)")
        accs_sym, forces_sym, var_syms, ext_syms = build_abkowitz_equations(
            coeffs, m_prime, mxg_prime, iz_prime)
        self._acc_func = sp.lambdify(var_syms + ext_syms, accs_sym, modules="numpy")
        self._force_func = sp.lambdify(var_syms, forces_sym, modules="numpy")
        self._prime_funcs = list(VARIABLES.values())

    @classmethod
    def from_yaml(cls, path: str, sd, eps: float, pod_forces=None):
        with open(path) as f:
            return cls(yaml.safe_load(f), sd, eps, pod_forces=pod_forces)

    def _primes(self, y, inp):
        U = max(np.sqrt(y[3]**2 + y[4]**2), self.eps)
        return [f(y, inp, U, self.sd) for f in self._prime_funcs], U

    def _q(self, U):
        # Normierung der Kraefte: 0.5*rho*L*D*U^2 (Moment zusaetzlich *L)
        return 0.5 * self.sd.rho * self.sd.L * self.sd.Tm * U**2

    def _pod_prime(self, t, y, inp, U):
        """Pod-Kraefte dimensional -> prime (X', Y', N')."""
        if self.pod_forces is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        pX, pY, pN = self.pod_forces(t, y, inp)
        q = self._q(U)
        return (pX / q, pY / q, pN / (q * self.sd.L)), (pX, pY, pN)

    def derivatives(self, t, y, inp):
        vals, U = self._primes(y, inp)
        ext, _ = self._pod_prime(t, y, inp, U)
        udot_p, vdot_p, rdot_p = np.asarray(self._acc_func(*vals, *ext)).ravel()
        L = self.sd.L
        psi = y[2]
        return np.array([
            y[3] * np.cos(psi) - y[4] * np.sin(psi),
            y[3] * np.sin(psi) + y[4] * np.cos(psi),
            y[5],
            udot_p * U**2 / L,
            vdot_p * U**2 / L,
            rdot_p * U**2 / L**2,
        ])

    def forces(self, t, y, inp):
        """Dimensionale Kraefte X, Y [N] und Moment N [Nm].

        Ohne Pod-Modul: (F_X, F_Y, M_N) = Rumpf.
        Mit Pod-Modul: (hull_*, pod_*, F_*) mit F = Rumpf + Pod.
        """
        vals, U = self._primes(y, inp)
        Xp, Yp, Np = np.asarray(self._force_func(*vals)).ravel()
        q = self._q(U)
        hull = (Xp * q, Yp * q, Np * q * self.sd.L)
        if self.pod_forces is None:
            return hull
        _, pod = self._pod_prime(t, y, inp, U)
        return (*hull, *pod,
                hull[0] + pod[0], hull[1] + pod[1], hull[2] + pod[2])


if __name__ == "__main__":
    # Smoke-Test mit der echten Koeffizientendatei und den echten Schiffsdaten
    from Krylov.krylov_data import ShipConfig

    with open("data/03_primary/wlfa/abkowitz_coefficients.yml") as f:
        coeffs = yaml.safe_load(f)
    with open("data/01_raw/wlfa/ship_data.yml") as f:
        sd = ShipConfig(**yaml.safe_load(f))

    ignored, rest = split_mass_entries(coeffs)
    m_prime, mxg_prime, iz_prime = mass_primes_from_ship(sd)
    print(f"m'={m_prime:.4f}, mxg'={mxg_prime:.5f}, Iz'={iz_prime:.5f}, ignoriert: {ignored}")
    accs, forces, syms, ext_syms = build_abkowitz_equations(rest, m_prime, mxg_prime, iz_prime)
    print("Kraefte X', Y', N':")
    sp.pprint(forces)
    print("Beschleunigungen udot', vdot', rdot':")
    sp.pprint(accs)
