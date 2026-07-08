# Regression — hydrodynamische Kraftkoeffizienten aus Kalman-Trials

`ForceRegression` ([regression.py](regression.py)) schaetzt Rumpf-Kraft-
koeffizienten (Abkowitz-Style oder ein eigenes Termeset) aus einem oder
mehreren gefilterten Trials. Die Idee: die Bewegungsgleichung wird
"rueckwaerts" benutzt — aus einer gemessenen Bewegung folgt die Kraft, die
sie erfordert hat; bekannte Anteile (Pod, Wind) werden abgezogen, der Rest
ist die zu regressierende Hydrodynamik.

```
Kalman_filter (kalman_filter.py)        ForceRegression (regression.py)
 └── filter(dt) -> DataFrame              ├── add_run(kalman_result, kf)  je Trial
     mit x0,y0,psi,u,v,r                  │     -> X_hydro, Y_hydro, N_hydro
     UND udot,vdot,rdot                   ├── build_dataset()  Trials aneinanderhaengen
                                          ├── fit(term_spec)  OLS je Gleichung
                                          └── to_yaml()  -> abkowitz_coefficients_regressed_*.yml
```

## Schnellstart

```python
from Krylov.kalman_filter import ExtendedKalmanFilter
from Krylov.krylov import Krylov_forces, KrylovModel
from Krylov.abkowitz import AbkowitzModel
from Krylov.regression import ForceRegression

# --- pro Trial: EKF laufen lassen (liefert jetzt auch udot,vdot,rdot) ---
kf_turn = Krylov_forces(ship_parameters=sp, krylov_parameters=kp,
                        ship_resistance=res, prop_openwater=pow_, data=turn_data)
ekf_turn = ExtendedKalmanFilter(data=kf_turn.data, model=KrylovModel(kf_turn),
                                state_columns=state_columns, meas_coulumns=meas_columns,
                                input_columns=input_columns, P_0=P_0, Q=Q, R=R, H=H)
result_turn = ekf_turn.filter(dt=1.0)          # x0,y0,psi,u,v,r,udot,vdot,rdot

kf_zz = Krylov_forces(..., data=zigzag_data)
ekf_zz = ExtendedKalmanFilter(data=kf_zz.data, model=KrylovModel(kf_zz), ...)
result_zz = ekf_zz.filter(dt=1.0)

# --- Regression: Kraftmodell fuer die Starrkoerperseite waehlen ---
model = AbkowitzModel.from_yaml("data/03_primary/wlfa/abkowitz_coefficients.yml",
                                kf_turn.sd, eps=0.5)
reg = ForceRegression(model=model, sd=kf_turn.sd, t_settle=30.0)
reg.add_run(result_turn, kf_turn, name="turning_circle")
reg.add_run(result_zz, kf_zz, name="zigzag")
reg.build_dataset()

# --- Formel: gleiches Term-Format wie die Abkowitz-YAML ---
term_spec = {
    "Xu":  [None, ["u"]],
    "Xvv": [None, ["v", "v"]],
    "Xrr": [None, ["r", "r"]],
    "Yv":  [None, ["v"]],
    "Yr":  [None, ["r"]],
    "Nv":  [None, ["v"]],
    "Nr":  [None, ["r"]],
}
results = reg.fit(term_spec)
print(reg.summaries()["X"])            # R^2, p-Werte, Konfidenzintervalle je Term
reg.to_yaml()                          # abkowitz_coefficients_regressed_<timestamp>.yml
```

Das exportierte YAML ist direkt mit `AbkowitzModel.from_yaml(...)` ladbar —
schliesst den Kreis zum [Simulator](SIMULATOR.md).

## Modell-Protokoll: `rigid_body_forces`

Damit `ForceRegression` modellagnostisch bleibt (Abkowitz, Krylov, oder ein
eigenes Modell — z.B. eine andere Bewegungsgleichung, die noch getestet
werden soll), braucht jedes Modell zusaetzlich zu `derivatives`/`forces`/
`force_columns` (siehe [SIMULATOR.md](SIMULATOR.md)) genau ein weiteres
Attribut:

| Attribut | Signatur | Bedeutung |
|---|---|---|
| `rigid_body_forces` | `(y, acc) -> (X, Y, N)` dimensional [N, N, Nm] | Kraft/Moment, die die gegebene Bewegung erfordert — reine Traegheitsseite der EOM, **ohne** hydrodynamische Kraft |

`y = [x0, y0, psi, u, v, r]`, `acc = [udot, vdot, rdot]` — beide dimensional,
aus der gemessenen/gefilterten Trajektorie (nicht aus einer Simulation!).

Jedes Modell implementiert das mit seiner **eigenen** Massenberechnung und
seiner **eigenen** Bewegungsgleichung — es gibt keine zweite/generische
Formel in `regression.py`, die dupliziert werden muesste:

- **`AbkowitzModel.rigid_body_forces`** ([abkowitz.py](abkowitz.py)): nutzt
  `rigid_body_lhs` (m', mxg', Iz' aus den Schiffsdaten via
  `mass_primes_from_ship`) — dieselbe Formel, die auch beim Vorwaerts-Loesen
  nach den Beschleunigungen (`build_abkowitz_equations`) verwendet wird.
  Dimensionalisiert mit `q_scale` (auch von `AbkowitzModel.forces`
  verwendet).
- **`KrylovModel.rigid_body_forces`** ([krylov.py](krylov.py)): loest
  `Krylov_forces.equations()` symbolisch nach `F_X, F_Y, M_N` auf (statt
  nach den Beschleunigungen) — `Krylov_forces.rigid_body_force_func(name=...)`.
  Nutzt `hydro_mass_dict` (hydrodynamische Masse, Munk-Ellipsoid) + `sd.m`,
  komplett dimensional (kein Prime-System).
- **Ein eigenes Modell** (z.B. eine neue Bewegungsgleichung): einfach
  `rigid_body_forces(y, acc)` auf dem Modell-Objekt implementieren, am besten
  ebenfalls durch Invertieren der bereits vorhandenen symbolischen EOM
  (`sp.solve` nach den Kraftsymbolen statt nach den Beschleunigungen) —
  keine zweite Formel von Hand schreiben.

## Datenfluss in `add_run`

Pro Trial (ein EKF-Ergebnis + die zugehoerige `Krylov_forces`-Instanz `kf`):

1. Erste `t_settle` Sekunden verwerfen (EKF-Einschwingphase — `P` ist am
   Trial-Anfang noch gross, die gefilterten Zustaende sind dort unzuverlaessig).
2. Input (`N0, N1, delta_r0, delta_r1`) vom Rohdaten-Zeitraster (`kf.data`)
   auf das EKF-Zeitraster interpolieren (Zero-Order-Hold, wie in
   [simulator.py](simulator.py)).
3. `model.rigid_body_forces(y, acc)` -> `X_total, Y_total, N_total`.
4. `kf.calc_pod_forces` und `kf.calc_windforces` an derselben Zeile
   auswerten und abziehen:
   `X_hydro = X_total - pod_X - wind_X` (analog Y, N).
5. Alle Zwischenspalten (`X_total`, `pod_X`, `wind_X`, `X_hydro`, ...) landen
   im DataFrame — volle Nachvollziehbarkeit, gleiches Muster wie
   `KrylovModel.force_columns`.

`build_dataset()` haengt alle Trials mit `pd.concat(..., ignore_index=True)`
zusammen — **kein** Sortieren nach Zeit: der Zustand wird als Markov-Zustand
behandelt, die Reihenfolge zwischen (und innerhalb der) Trials ist fuer die
Regression irrelevant.

## Beschleunigungen (`ExtendedKalmanFilter.filter`)

`filter()` haengt an `u, v, r` automatisch `udot, vdot, rdot` an:
Savitzky-Golay-Glaettung (`savgol_window`, `savgol_polyorder`) gegen das
Rauschen im EKF-Output, danach zentrale Differenzen (`np.gradient(., dt)`).
Reine Finite-Differenzen ohne Glaettung wuerden das Messrauschen verstaerken
und als Fehler-in-Variablen-Problem direkt in die Regression einfliessen.

## Fit (`fit`)

Term-Spec im selben Format wie die Abkowitz-Koeffizienten-YAML:
`Name: [wert, [variablen]]`, der erste Buchstabe des Namens (`X`/`Y`/`N`)
entscheidet die Gleichung, `wert` wird ignoriert (die Regression bestimmt
ihn). Die Terme werden mit den **gleichen** `VARIABLES`-Funktionen aus
[abkowitz.py](abkowitz.py) ausgewertet wie im Simulator — eine neue Variable
dort ergaenzen reicht fuer beide Seiten.

**Modell-Protokoll: `known_forces`** (optional, Default `("pod", "wind")`):
`fit` zieht von `X_total`/`Y_total`/`N_total` genau die Fremdkraefte ab, die
im `known_forces`-Tupel des Modells stehen — der Rest bleibt im Residuum und
wird gegen die `term_spec`-Terme gefittet. `AbkowitzModel` und `KrylovModel`
setzen beide standardmaessig `("pod", "wind")` (bisheriges Verhalten,
unveraendert: Pod- und Windkraft gelten als bekannt). Ein "volles"
Abkowitz-Modell, das Pod-/Ruderkraefte selbst ueber die `delta`-Variable
mitregressieren soll (statt sich auf `kf.calc_pod_forces` zu verlassen),
setzt `known_forces=("wind",)` — z.B.
`AbkowitzModel.from_yaml(..., known_forces=("wind",))` fuer `abkowitz_full`;
dann bleibt nur die Windkraft abgezogen, die Pod-Kraft steckt weiter im
Residuum. Fehlt das Attribut ganz (eigenes Modell ohne `known_forces`), gilt
`DEFAULT_KNOWN_FORCES = ("pod", "wind")` aus `regression.py`.

Pro Gleichung ein `statsmodels.OLS` im Prime-System (`q_scale`-Normierung,
gleiche Konvention wie `AbkowitzModel`). Zeilen mit `U <= eps` werden
ausgeschlossen (Division im Prime-System instabil bei ~Stillstand).
`summaries()` gibt die `statsmodels`-Summary (R², p-Werte,
Konfidenzintervalle) je Gleichung zurueck — vor dem Export pruefen!

**Added mass (`Xudot`, `Yvdot`, `Yrdot`, `Nvdot`, `Nrdot`):** `variable_values`
bietet neben `u,v,r,delta` auch `udot,vdot,rdot` (primed, aus den gemessenen
Beschleunigungen) als Regressor-Variablen an. Physikalisch ist das die
Reaktionskraft des Wassers auf die Beschleunigung — eine hydrodynamische
Kraft, keine Schiffs-Massenzunahme. `rigid_body_forces` verwendet deshalb nur
die reale Schiffsmasse (`m', mxg', Iz'` aus den Schiffsdaten) und rechnet
diese Terme **nicht** heraus; sie stecken in `X_hydro/Y_hydro/N_hydro` und
koennen ganz normal per `term_spec` (z.B. `"Xudot": [None, ["udot"]]`)
mitgefittet werden — konsistent mit der bestehenden
`abkowitz_coefficients.yml`, die genau diese Terme schon enthaelt.

## Export (`to_yaml`)

Schreibt die gefitteten Koeffizienten im `[wert, [variablen]]`-Format;
Dateiname per Default `abkowitz_coefficients_regressed_<timestamp>.yml`
(Kopfzeile markiert die Datei explizit als regressiert, nicht gemessen) —
damit nie versehentlich mit einer manuell kuratierten Koeffizientendatei
verwechselt. Direkt ladbar via `AbkowitzModel.from_yaml(...)`.
