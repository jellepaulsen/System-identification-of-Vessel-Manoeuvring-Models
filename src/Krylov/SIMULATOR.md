# Simulator — modell-agnostische Manövriersimulation

Der Simulationscode (Integration, Zero-Order-Hold der Inputs, Zigzag-Logik,
Ergebnis-DataFrames) ist vom Kraftmodell getrennt. Es gibt genau einen
Einstiegspunkt: die Klasse `Simulator` in [simulator.py](simulator.py).
Kraftmodelle werden per Name registriert und bei der Simulation ausgewählt —
so lassen sich Krylov, Abkowitz (und später weitere Modelle) auf identischen
Inputs rechnen und untereinander bzw. mit Echtzeit-Messdaten vergleichen.

```
Simulator (simulator.py)                   Modelle
 ├── data: Inputs (Index = Zeit [s])        ├── KrylovModel   (krylov.py)
 ├── add_model(name, model) / sim[name]     ├── AbkowitzModel (abkowitz.py)
 ├── simulate(name, x0)                     └── <eigenes Modell>
 └── simulate_zigzag(name, x0, cfg)
```

## Schnellstart

```python
from Krylov.krylov import Krylov_forces, KrylovModel
from Krylov.abkowitz import AbkowitzModel
from Krylov.simulator import Simulator, ZigzagConfig

kf = Krylov_forces(ship_parameters=..., krylov_parameters=...,
                   ship_resistance=..., prop_openwater=..., data=input_data)

sim = Simulator(data=input_data)   # DataFrame mit N0, N1, delta_r0, delta_r1; Index = Zeit [s]
sim.add_model("krylov", KrylovModel(kf))
sim.add_model("abkowitz", AbkowitzModel.from_yaml(
    "data/03_primary/wlfa/abkowitz_coefficients.yml", kf.sd, kf.eps))

x0 = [0, 0, 0, 3.6, 0, 0]          # [x0, y0, psi, u, v, r]
df_kry = sim.simulate("krylov", x0)
df_abk = sim.simulate("abkowitz", x0)   # gleiche Inputs -> direkt vergleichbar

# Abkowitz modular (wie Krylov/MMG): Rumpfpolynom + Pod-/Ruderkraefte.
# Die Pod-Kraefte kommen aus calc_pod_rud_force der Krylov_forces-Instanz.
# Auswahl beim Simulationsstart ueber den Modellnamen:
from Krylov.abkowitz import pod_forces_from_krylov
sim.add_model("abkowitz_pod", AbkowitzModel.from_yaml(
    "data/03_primary/wlfa/abkowitz_coefficients.yml", kf.sd, kf.eps,
    pod_forces=pod_forces_from_krylov(kf)))
df_pod = sim.simulate("abkowitz_pod", x0)

zz = sim.simulate_zigzag("abkowitz", x0,
        ZigzagConfig(N=190, delta_deg=10, psi_des=10, n_switches=4, t0=60, t_aft=120))
```

Beide `simulate`-Aufrufe liefern ein DataFrame mit identischem Grundlayout:
Zustände (`x0, y0, psi, u, v, r`), Inputs (`N0, N1, delta_r0, delta_r1`) und
den Kraftspalten des Modells — `F_X, F_Y, M_N` gibt es bei jedem Modell,
`KrylovModel` liefert zusätzlich `kr_*`, `pod_*` und `cns`; `AbkowitzModel`
mit Pod-Modul liefert `hull_*`, `pod_*` und `F_*` (= hull + pod).
`simulate_zigzag` hängt eine `phase`-Spalte an (`accel`/`maneuver`/`coast`).

**Abkowitz mit/ohne Pod-Kräfte:** ohne `pod_forces` ist das Modell ein reines
Rumpfpolynom — alles (auch die Ruderwirkung) muss in den YAML-Koeffizienten
stecken. Mit `pod_forces=pod_forces_from_krylov(kf)` wird es modular: die
YAML sollte dann nur Rumpfderivative enthalten (delta-Terme null lassen!),
Schub und Ruderkraft kommen aus dem Pod-Modul. Die dimensionalen Pod-Kräfte
werden intern mit ½ρLDU² (bzw. ·L) entdimensionalisiert und gehen als
X'_ext/Y'_ext/N'_ext in die symbolisch gelösten Bewegungsgleichungen ein.

## Vergleich mit Messdaten

`KrylovComparison` ([comparison_tuning.py](comparison_tuning.py)) hält den
Simulator unter `comp.simulator` und rechnet standardmäßig `"krylov"`:

```python
comp = KrylovComparison(kf, trial_parquet)
comp.prepare_inputs()                    # Messkanäle -> comp.simulator.data
comp.run()                               # model="krylov"
comp.simulator.add_model("abkowitz", AbkowitzModel.from_yaml(path, kf.sd, kf.eps))
comp.run(model="abkowitz")               # gleiches Trial, anderes Modell
print(comp.error_factors())
```

## Modell-Interface

Ein Modell ist ein beliebiges Objekt mit drei Attributen (Duck-Typing, keine
Basisklasse nötig). `add_model` prüft das beim Registrieren:

| Attribut | Signatur | Bedeutung |
|---|---|---|
| `derivatives` | `(t, y, inp) -> array(6)` | Zeitableitung `[ẋ0, ẏ0, ψ̇, u̇, v̇, ṙ]` |
| `forces` | `(t, y, inp) -> tuple` | Kraftwerte für das Ausgabe-DataFrame |
| `force_columns` | `list[str]` | Spaltennamen dazu; `F_X, F_Y, M_N` immer enthalten |

Dabei ist `y` der Zustand `[x0, y0, psi, u, v, r]` (SI, Winkel in rad) und
`inp` die aktuelle Input-Zeile `[N0, N1, delta_r0, delta_r1]`. Der Simulator
kümmert sich um Zero-Order-Hold (`simulate`) bzw. stückweise konstante Inputs
(`simulate_zigzag`) — das Modell sieht immer nur die aktuelle Zeile.

Ein neues Modell anbinden heißt also: die drei Attribute implementieren und
`sim.add_model("meinmodell", MeinModell(...))` aufrufen. Fertig.

## Abkowitz-Koeffizienten (YAML)

Datei: [data/03_primary/wlfa/abkowitz_coefficients.yml](../../data/03_primary/wlfa/abkowitz_coefficients.yml)
— flaches Dict, ein Koeffizient pro Zeile:

```yaml
Xuu:   [-0.0324, [u, u]]     # Beitrag = -0.0324 * u' * u'
Y0:    [-0.0018, []]         # leere Liste = konstanter Bias-Term
Yvdot: [-0.1359, [vdot]]     # Beschleunigungsterm -> Massenmatrix (LHS)
Nd:    [null,    [delta]]    # null = Koeffizient wird ignoriert
```

Regeln:
- Erster Buchstabe des Namens wählt die Gleichung (X / Y / N)
- Multiplikator = Liste von Variablennamen; Potenzen durch Wiederholung (`[u, u]` = u'²)
- Erlaubte Variablen: `u, v, r, delta` plus `udot, vdot, rdot` für Massenterme.
  Erweiterbar über die `VARIABLES`-Registry in [abkowitz.py](abkowitz.py)
  (eine Zeile pro neuer Variable, z.B. Propellerdrehzahl `n`)
- **Alle Werte müssen echte Prime-Werte sein** (typisch Größenordnung 10⁻¹…10⁻⁴),
  keine Tabellen-Notation „×10³". Da Masse und Pod-Kräfte unabhängig von den
  Koeffizienten skaliert sind, führt eine falsche Skala zu einem viel zu trägen
  oder überdrehten Schiff
- `m'`, `mxg'`, `Iz'` stehen **nicht** in der YAML — sie werden aus den
  Schiffsdaten berechnet (`mass_primes_from_ship`: `m' = m/(½ρL²D)`,
  `Iz' = I_z/(½ρL⁴D)`, `mxg' = m·xg/(½ρL³D)` mit `xg = lcg − L/2`) und gehen
  mit den vollen xg-Kopplungstermen in die Bewegungsgleichungen ein. Noch
  vorhandene m/mxg/Iz-Einträge werden beim Laden ignoriert (mit Meldung)

### Prime-System (Normierung)

Alle YAML-Werte sind dimensionslos. Entdimensionalisierung mit L = `sd.L`,
D = Tiefgang `sd.Tm`, U = momentane Geschwindigkeit √(u²+v²) (mind. `eps`):

| Größe | Faktor | | Größe | Faktor |
|---|---|---|---|---|
| m, mxg | ½ρL²D | | u, v | U |
| Iz | ½ρL⁴D | | r | U/L |
| X, Y | ½ρLDU² | | u̇, v̇ | U²/L |
| N | ½ρL²DU² | | ṙ | U²/L² |

Achtung: das ist das **D-basierte** Prime-System (wie die Quelle der
wlfa-Koeffizienten). Das Paket `vessel_manoeuvring_models` (Kedro-Pipelines)
nutzt das L-basierte System (½ρL³, ½ρU²L², …) — Koeffizienten von dort müssen
vor Übernahme pro D-Potenz mit L/T umgerechnet werden.

`AbkowitzModel` rechnet intern: Zustand → Prime-Variablen (`u' = u/U`,
`r' = r·L/U`, `delta` = Mittel aus `delta_r0`/`delta_r1`), wertet die mit
sympy symbolisch nach `u̇', v̇', ṙ'` aufgelösten Gleichungen aus und
redimensionalisiert mit `U²/L` bzw. `U²/L²`. Beachte: mit `u' = u/U` ist bei
Geradeausfahrt `u' ≈ 1` — Literaturwerte, die mit Störgrößen `Δu = u − U₀`
definiert sind, müssen entsprechend interpretiert werden.

### Bewegungsgleichungen (Prime, inkl. xg-Kopplung)

```
m'·(u̇' − v'r') − mxg'·r'²        = X'(u', v', r', δ) + Xudot'·u̇'
m'·(v̇' + u'r') + mxg'·ṙ'         = Y'(...) + Yvdot'·v̇' + Yrdot'·ṙ'
Iz'·ṙ' + mxg'·(v̇' + u'r')        = N'(...) + Nvdot'·v̇' + Nrdot'·ṙ'
```

sympy löst dieses lineare System einmal symbolisch nach den Beschleunigungen
auf (`build_abkowitz_equations`), danach ist die rhs eine reine
numpy-Auswertung (lambdify).
