import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from datetime import datetime


def calculate_sog(
    df: pd.DataFrame,
    source: str = "gps",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    ax_col: str = "linear_acceleration.x",
    ay_col: str = "linear_acceleration.y",
) -> pd.Series:
    """Berechnet Speed over Ground in m/s aus GPS oder IMU."""
    sog = pd.Series(np.nan, index=df.index, name="sog_mps")

    if source == "gps":
        data = df[["timestamp_ns", lat_col, lon_col]].dropna().sort_values("timestamp_ns")
        if data.empty:
            return sog
        lat = np.radians(data[lat_col].to_numpy())
        lon = np.radians(data[lon_col].to_numpy())

        dlat = np.diff(lat)
        dlon = np.diff(lon)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
        a = np.clip(a, 0, 1)
        distance = 2 * 6_371_000 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dt = np.diff(data["timestamp_ns"].to_numpy()) / 1e9
        speed = np.divide(distance, dt, out=np.full_like(distance, np.nan), where=dt > 0)
        sog.loc[data.index] = np.r_[np.nan, speed]

    elif source == "imu":
        data = df[["timestamp_ns", ax_col, ay_col]].dropna().sort_values("timestamp_ns")
        if data.empty:
            return sog
        dt = np.diff(data["timestamp_ns"].to_numpy(), prepend=data["timestamp_ns"].iloc[0]) / 1e9
        vx = np.cumsum(data[ax_col].to_numpy() * dt)
        vy = np.cumsum(data[ay_col].to_numpy() * dt)
        sog.loc[data.index] = np.hypot(vx, vy)

    else:
        raise ValueError("source muss 'gps' oder 'imu' sein")

    return sog


class TelemetryPlotter:
    DATE_FMT = "%d.%m.%Y:%H:%M:%S"
    
    # Globales Farb- und Styling-Dictionary für Spalten
    # STB (Steuerbord/rechts) = Grün | PS (Backbord/links) = Rot
    COLUMN_STYLE = {
        # Azimuth Commands
        "Pod_azimuth_stb_cmd": {"color": "#3fc53f", "linestyle": "-.", "linewidth": 1.5},
        "Pod_azimuth_ps_cmd": {"color": "#ec4545", "linestyle": "-.", "linewidth": 1.5},
        
        # Azimuth Response
        "azimuth_response_stb": {"color": "#1f7714", "linestyle": "-", "linewidth": 1.8},
        "azimuth_response_ps": {"color": "#b81c1c", "linestyle": "-", "linewidth": 1.8},
        
        # Azimuth Response Gradient
        "azimuth_response_stb_grad": {"color": "#7fbf7f", "linestyle": "--", "linewidth": 1.2},
        "azimuth_response_ps_grad": {"color": "#ff7f7f", "linestyle": "--", "linewidth": 1.2},
        
        # Azimuth Offset (optional)
        "azimuth_offset_stb": {"color": "#98df8a", "linestyle": ":", "linewidth": 1.0},
        "azimuth_offset_ps": {"color": "#ffbb78", "linestyle": ":", "linewidth": 1.0},
        
        # Heading & Rate of Turn
        "heading": {"color": "#9467bd", "linestyle": "-", "linewidth": 2.0},
        "rate_of_turn": {"color": "#c5b0d5", "linestyle": "-", "linewidth": 1.5},
    }
    
    @staticmethod
    def clamp(arr, min_val=None, max_val=None):
        """
        Clamp (beschränkt) die Werte eines Arrays/Series auf [min_val, max_val].
        Wenn min_val oder max_val None ist, bleibt diese Grenze offen.
        """
        arr = np.asarray(arr)
        if min_val is not None:
            arr = np.maximum(arr, min_val)
        if max_val is not None:
            arr = np.minimum(arr, max_val)
        return arr
    
    

    def __init__(self, df: pd.DataFrame):
        """
        df must have a 'timestamp_ns' column in nanoseconds (Unix epoch)
        and latitude/longitude columns.
        """
        self.df = df.copy()
        self.df["_dt"] = pd.to_datetime(self.df["timestamp_ns"], unit="ns", utc=True)
        self.df["_dt_local"] = self.df["_dt"].dt.tz_convert("Europe/Amsterdam")
        
    def _to_utc(self, ts_str: str) -> str:
        """
        Konvertiert einen Zeitstempel-String im Format '%d.%m.%Y:%H:%M:%S' von UTC+2 nach UTC.
        Gibt einen String im gleichen Format zurück, aber in UTC.
        """
        dt = datetime.strptime(ts_str, self.DATE_FMT)
        dt_utc = dt - pd.Timedelta(hours=2)
        result = str(dt_utc.strftime(self.DATE_FMT))
        # print(f"UTC+2: {ts_str}  →  UTC: {result}")
        return result


    def _parse_ts(self, ts_str: str) -> pd.Timestamp:
        # Konvertiere von UTC+2 nach UTC
        ts_str_utc = self._to_utc(ts_str)
        dt = datetime.strptime(ts_str_utc, self.DATE_FMT)
        return pd.Timestamp(dt, tz="UTC")

    def _filter_window(self, start: str, end: str) -> pd.DataFrame:
        t0 = self._parse_ts(start)
        t1 = self._parse_ts(end)
        mask = (self.df["_dt"] >= t0) & (self.df["_dt"] <= t1)
        return self.df.loc[mask]

    def _clean_nans(self, df_subset: pd.DataFrame, cols: list | str) -> pd.DataFrame:
        """
        Bereinigt das DataFrame um NaN-Werte für eine oder mehrere spezifische Spalten.
        Dadurch können durchgängige Linien zwischen den echten Werten gezogen werden,
        ohne dass Matplotlib durch NaNs unterbrochen wird.
        """
        if isinstance(cols, str):
            cols = [cols]
        return df_subset.dropna(subset=cols)

    @staticmethod
    def _disable_scientific(ax, axis: str = "both") -> None:
        """Force plain numeric ticks without scientific notation or offset."""
        if axis in ("both", "x"):
            xfmt = ScalarFormatter(useOffset=False)
            xfmt.set_scientific(False)
            ax.xaxis.set_major_formatter(xfmt)
        if axis in ("both", "y"):
            yfmt = ScalarFormatter(useOffset=False)
            yfmt.set_scientific(False)
            ax.yaxis.set_major_formatter(yfmt)

    def stats_for_window(self, start: str, end: str, column: str | list, figsize: tuple = (10, 4), plot: bool = True) -> pd.DataFrame:
        """
        Berechnet Mittelwert, Standardabweichung und weitere Stats für Spalte(n) im Zeitfenster.
        
        Parameters
        ----------
        start : str – e.g. "22.04.2026:08:00:00"
        end : str – e.g. "22.04.2026:09:30:00"
        column : str or list – Spaltenname oder Liste von Spaltennamen
        figsize : tuple – Größe der Tabelle (width, height)
        plot : bool – Wenn True, wird die Tabelle geplottet
        
        Returns
        -------
        pd.DataFrame – Statistik-Tabelle mit Zeilen für jede Spalte
        """
        subset = self._filter_window(start, end)
        
        if subset.empty:
            print(f"No data in window {start} – {end}")
            return pd.DataFrame()
        
        # Stelle sicher, dass column eine Liste ist
        if isinstance(column, str):
            columns = [column]
        else:
            columns = list(column)
        
        stats_list = []
        for col in columns:
            if col not in subset.columns:
                print(f"Warning: Column '{col}' not found in DataFrame.")
                continue
            
            valid_data = self._clean_nans(subset, col)
            
            if valid_data.empty:
                print(f"Warning: Only NaN values for '{col}' in given window.")
                stats_list.append({
                    "Column": col,
                    "Mean": None,
                    "Std": None,
                    "Count": 0,
                    "Min": None,
                    "Max": None,
                })
                continue
            
            values = valid_data[col]
            stats_list.append({
                "Column": col,
                "Mean": values.mean(),
                "Std": values.std(),
                "Count": len(values),
                "Min": values.min(),
                "Max": values.max(),
            })
        
        stats_df = pd.DataFrame(stats_list)
        
        if plot and not stats_df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.axis("tight")
            ax.axis("off")
            
            # Formatiere Zahlen für die Anzeige
            table_data = []
            for _, row in stats_df.iterrows():
                table_data.append([
                    row["Column"],
                    f"{row['Mean']:.4f}" if pd.notna(row['Mean']) else "N/A",
                    f"{row['Std']:.4f}" if pd.notna(row['Std']) else "N/A",
                    f"{row['Count']}" if row['Count'] > 0 else "0",
                    f"{row['Min']:.4f}" if pd.notna(row['Min']) else "N/A",
                    f"{row['Max']:.4f}" if pd.notna(row['Max']) else "N/A",
                ])
            
            table = ax.table(
                cellText=table_data,
                colLabels=["Column", "Mean", "Std", "Count", "Min", "Max"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Header formatieren
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor("#40466e")
                table[(0, i)].set_text_props(weight="bold", color="white")
            
            # Alternating row colors
            for i in range(1, len(table_data) + 1):
                for j in range(len(table_data[0])):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor("#f0f0f0")
                    else:
                        table[(i, j)].set_facecolor("white")
            
            ax.set_title(f"Statistics  {start}  →  {end}", fontsize=12, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.show()
        
        return stats_df

    def plot_track(
        self,
        start: str = "01.01.1970:00:00:00",
        end: str = "31.12.2100:23:59:59",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        figsize: tuple = (12, 8),
        invert_y: bool = False,
    ) -> None:
        """
        Plot latitude vs longitude for the given time window.

        Parameters
        ----------
        start : str  – e.g. "22.04.2026:08:00:00"
        end   : str  – e.g. "22.04.2026:09:30:00"
        lat_col, lon_col : column names in the DataFrame
        figsize : tuple – tuple specifying the figure size (width, height)
        """
        subset = self._filter_window(start, end)
        print(f"start unix: {subset['timestamp_ns'].min()}  →  {subset['_dt'].min()}")
        if subset.empty:
            print(f"No data in window {start} – {end}")
            return

        valid_data = self._clean_nans(subset, [lat_col, lon_col])
        if valid_data.empty:
            print(f"No valid coordinate data in window {start} – {end}")
            return

        t = valid_data["timestamp_ns"].astype(float)
        t_norm = (t - t.min()) / (t.max() - t.min()) if t.max() > t.min() else t * 0

        _, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(
            valid_data[lon_col],
            valid_data[lat_col],
            c=t_norm,
            cmap="Blues",
            vmin=-0.2,
            s=4,
        )
        plt.colorbar(sc, ax=ax, label="Time (early → late)")
        ax.set_xlabel(f"Longitude  [{lon_col}]")
        ax.set_ylabel(f"Latitude  [{lat_col}]")
        ax.set_title(f"Track  {start}  →  {end}")
        ax.set_aspect("equal")
        if invert_y:
            ax.invert_yaxis()

        self._disable_scientific(ax, axis="both")
        plt.tight_layout()
        plt.show()

    def plot_time_series(
        self,
        start: str,
        end: str,
        columns: list,
        figsize: tuple = (10, 3),
        invert_y: bool = False,
        clamp_min: float = None,
        clamp_max: float = None,
    ) -> None:
        """
        Plot various data columns over time for the given time window.

        Parameters
        ----------
        start : str  – e.g. "22.04.2026:08:00:00"
        end   : str  – e.g. "22.04.2026:09:30:00"
        columns : list – list of column names in the DataFrame to plot
        figsize : tuple – tuple specifying the figure size (width, height), default 10:3
        """
        subset = self._filter_window(start, end)

        if subset.empty:
            print(f"No data in window {start} – {end}")
            return

        _, ax = plt.subplots(figsize=figsize)
        
        for col in columns:
            if col in subset.columns:
                valid_data = self._clean_nans(subset, col)
                if not valid_data.empty:
                    y = valid_data[col]
                    y = self.clamp(y, clamp_min, clamp_max)
                    
                    # Verwende Styling aus COLUMN_STYLE Dictionary
                    style = self.COLUMN_STYLE.get(col, {})
                    ax.plot(valid_data["_dt_local"], y, label=col, **style)
                else:
                    print(f"Warning: Only NaN values for '{col}' in given window.")
            else:
                print(f"Warning: Column '{col}' not found in DataFrame.")

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"Time Series  {start}  →  {end}")
        ax.legend()
        if invert_y:
            ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_auto(
        self,
        start: str,
        end: str,
        columns: list,
        twin_axis: bool = False,
        figsize: tuple = (12, 4),
        title: str = "",
        invert_y: bool = False,
        clamp_min: float = None,
        clamp_max: float = None,
    ) -> None:
        subset = self._filter_window(start, end)

        if subset.empty:
            print(f"No data in window {start} – {end}")
            return

        fig, ax = plt.subplots(figsize=figsize)

        if twin_axis:
            main_cols = columns[:-1]
            twin_col = columns[-1]
        else:
            main_cols = columns
            twin_col = None

        for col in main_cols:
            if col not in subset.columns:
                print(f"Warning: Column '{col}' not found.")
                continue
            valid = self._clean_nans(subset, col)
            if valid.empty:
                print(f"Warning: Only NaN values for '{col}'.")
                continue
            y = self.clamp(valid[col], clamp_min, clamp_max)
            
            # Verwende Styling aus COLUMN_STYLE Dictionary
            style = self.COLUMN_STYLE.get(col, {})
            ax.plot(valid["_dt_local"], y, label=col, **style)

        ax.set_xlabel("Time")
        ax.set_ylabel(", ".join(main_cols))
        ax.set_title(title or f"{', '.join(columns)}  {start}  →  {end}")

        handles, labels = ax.get_legend_handles_labels()

        if twin_col:
            if twin_col not in subset.columns:
                print(f"Warning: Column '{twin_col}' not found.")
            else:
                valid = self._clean_nans(subset, twin_col)
                if valid.empty:
                    print(f"Warning: Only NaN values for '{twin_col}'.")
                else:
                    y = self.clamp(valid[twin_col], clamp_min, clamp_max)
                    ax2 = ax.twinx()
                    # Verwende Styling aus COLUMN_STYLE Dictionary
                    style = self.COLUMN_STYLE.get(twin_col, {"linestyle": "--", "color": "black"})
                    ax2.plot(valid["_dt_local"], y, label=twin_col, **style)
                    ax2.set_ylabel(twin_col)
                    h2, l2 = ax2.get_legend_handles_labels()
                    handles += h2
                    labels += l2

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz="Europe/Amsterdam"))
        ax.legend(handles, labels, loc="upper left")
        if invert_y:
            ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_azimuth_360(
        self,
        start: str,
        end: str,
        columns: list,
        figsize: tuple = (12, 4),
        title: str = "Azimuth 0-360",
        scatter: bool = False,
        clamp_min: float = None,
        clamp_max: float = None,
    ) -> None:
        subset = self._filter_window(start, end)
        if subset.empty:
            print(f"No data in window {start} – {end}")
            return

        fig, ax = plt.subplots(figsize=figsize)
        for col in columns:
            if col not in subset.columns:
                print(f"Warning: Column '{col}' not found in DataFrame.")
                continue

            valid_data = self._clean_nans(subset, col)
            if valid_data.empty:
                print(f"Warning: Only NaN values for '{col}' in given window.")
                continue

            y = self.clamp(valid_data[col], clamp_min, clamp_max)
            
            # Verwende Styling aus COLUMN_STYLE Dictionary
            style = self.COLUMN_STYLE.get(col, {})
            if scatter:
                ax.scatter(valid_data["_dt_local"], y, label=col, s=10, alpha=0.8, **{k: v for k, v in style.items() if k in ['color']})
            else:
                ax.plot(valid_data["_dt_local"], y, label=col, **style)

        ax.set_xlabel("Time")
        ax.set_ylabel("Azimuth [°]")
        ax.set_title(title or f"{', '.join(columns)}  {start}  →  {end}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz="Europe/Amsterdam"))
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _map_angle_to_180(self, angle_series: pd.Series, zero_point: float) -> pd.Series:
        """
        Mappt einen Winkel im Bereich [0, 360) so um, dass er im Bereich [-180, 180] liegt,
        wobei 'zero_point' als Nullpunkt (0) definiert wird.
        Winkel links von zero_point werden negativ, Winkel rechts davon positiv.
        """
        # Alle Werte relativ zum gewünschten Nullpunkt berechnen
        shifted = angle_series - zero_point
        # In den Bereich (-180, 180] normalisieren
        return (shifted + 180) % 360 - 180

    def plot_azimuth(
        self,
        start: str,
        end: str,
        columns: list,
        zero_point: float = 0.0,
        heading_col: str | None = None,
        zero_heading: float = 0.0,
        rot_col: str | None = None,
        rot_scale: float = 1.0,
        rot_clamp_min: float = None,
        rot_clamp_max: float = None,
        wind_speed_col: str | None = None,
        wind_dir_col: str | None = None,
        figsize: tuple = (12, 4),
        title: str = "Manoeuvring Telemetry",
        scatter: bool = False,
        clamp_min: float = None,
        clamp_max: float = None,
        invert_y: bool = False,
    ) -> None:
        """
        Plot azimuth / rudder data im Bereich [-180, 180] sowie optional Heading Change und ROT.

        Parameters
        ----------
        zero_point : float – Ruder-Nullpunkt [0, 360).
        heading_col : str – Spalte mit den Heading-Daten (z.B. "heading").
        zero_heading : float – Schiff-Nullpunkt/Sollkurs [0, 360). Heading Change wird dazu relativ berechnet.
        rot_col : str – Spalte mit der Rate of Turn (ROT). Bekommt eine sekundäre Y-Achse.
        rot_scale : float – Skalierungsfaktor für die ROT-Achse (default 1.0). Größer = höhere Auflösung.
        rot_clamp_min, rot_clamp_max : float – Begrenzungen nur für die ROT-Achse.
        wind_speed_col: str - Spalte für True Wind Speed
        wind_dir_col: str - Spalte für True Wind Direction
        """
        subset = self._filter_window(start, end)

        if subset.empty:
            print(f"No data in window {start} – {end}")
            return

        fig, ax = plt.subplots(figsize=figsize)

        def _plot_series(axis, x_values, y_values, label, *, color=None, linestyle="-", linewidth=1.5):
            if scatter:
                axis.scatter(x_values, y_values, label=label, s=10, alpha=0.8, color=color)
            else:
                axis.plot(x_values, y_values, label=label, linestyle=linestyle, linewidth=linewidth, color=color)
        
        # 1. Plot Azimuth/Rudder
        for col in columns:
            if col in subset.columns:
                valid_data = self._clean_nans(subset, col).copy()
                if not valid_data.empty:
                    # Ruderwinkel auf Bereich [-180, 180] relativ zum Nullpunkt mappen
                    valid_data[col] = ((valid_data[col] - zero_point + 180) % 360) - 180
                    y = self.clamp(valid_data[col], clamp_min, clamp_max)
                    
                    # Verwende Styling aus COLUMN_STYLE Dictionary, oder Defaults
                    style = self.COLUMN_STYLE.get(col, {"linestyle": "-", "linewidth": 1.5})
                    _plot_series(ax, valid_data["_dt_local"], y, label=f"{col}", **style)
                else:
                    print(f"Warning: Only NaN values for '{col}' in given window.")
            else:
                print(f"Warning: Column '{col}' not found in DataFrame.")

        # 2. Plot Heading Change
        if heading_col and heading_col in subset.columns:
            valid_heading = self._clean_nans(subset, heading_col).copy()
            if not valid_heading.empty:
                valid_heading[heading_col] = self._map_angle_to_180(valid_heading[heading_col], zero_heading)
                y = self.clamp(valid_heading[heading_col], clamp_min, clamp_max)
                # Verwende Styling aus COLUMN_STYLE Dictionary
                style = self.COLUMN_STYLE.get(heading_col, {"linestyle": "--", "linewidtfh": 2.0})
                _plot_series(
                    ax,
                    valid_heading["_dt_local"],
                    y,
                    label=f"Heading Change ({heading_col})",
                    **style
                )
            else:
                print(f"Warning: Only NaN values for '{heading_col}' in given window.")
        elif heading_col:
            print(f"Warning: Column '{heading_col}' not found in DataFrame.")


        ax.grid(True, which='both', axis='y',  alpha=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel("Angle [°] (Azimuth & Heading)")
        ax.set_title(f"{title}  {start}  →  {end}")
        
        # Dynamische Skalierung basierend auf der Zeitfenster-Länge (in Minuten)
        duration_mins = (self._parse_ts(end) - self._parse_ts(start)).total_seconds() / 60.0
        if duration_mins <= 3:
            ax.xaxis.set_major_locator(mdates.SecondLocator(bysecond=[0, 30]))
        elif duration_mins <= 5:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        elif duration_mins <= 15:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        elif duration_mins <= 30:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        elif duration_mins <= 60:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        else:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz="Europe/Amsterdam"))
        
        # 3. Plot ROT on secondary axis
        ax2 = None  # Wird nur erstellt wenn rot_col vorhanden
        if rot_col and rot_col in subset.columns:
            valid_rot = self._clean_nans(subset, rot_col).copy()
            if not valid_rot.empty:
                y_rot = self.clamp(valid_rot[rot_col], rot_clamp_min, rot_clamp_max)
                ax2 = ax.twinx()
                # Verwende Styling aus COLUMN_STYLE Dictionary
                style = self.COLUMN_STYLE.get(rot_col, {"color": "black", "linestyle": ":"})
                _plot_series(ax2, valid_rot["_dt_local"], y_rot, label=f"ROT ({rot_col})", **style)
                ax2.set_ylabel("Rate of Turn [°/min]")
                
                # Synchronisiere Nullpunkte zwischen Hauptachse und ROT-Achse
                ax_min, ax_max = ax.get_ylim()
                rot_min, rot_max = ax2.get_ylim()
                
                # Mache beide Achsen symmetrisch um Null (wichtig für gleiche Nullpunkte)
                ax_abs_max = max(abs(ax_min), abs(ax_max))
                ax.set_ylim(-ax_abs_max, ax_abs_max)
                
                # Berechne die neue ROT-Achsen-Skalierung mit rot_scale
                rot_range = max(abs(rot_min), abs(rot_max))
                
                if rot_range > 0:
                    # Skalierung: rot_scale bestimmt das Verhältnis zwischen ROT und Azimuth Wertebereichen
                    scale_factor = (ax_abs_max / rot_range) * rot_scale
                    ax2.set_ylim(-rot_range * scale_factor, rot_range * scale_factor)
                else:
                    # Fallback wenn keine ROT-Daten vorhanden
                    ax2.set_ylim(-ax_abs_max, ax_abs_max)
                    
                # ax2.legend(loc="upper right")
            else:
                print(f"Warning: Only NaN values for '{rot_col}' in given window.")
        elif rot_col:
            print(f"Warning: Column '{rot_col}' not found in DataFrame.")

        #combine legends from both axes
        handles, labels = ax.get_legend_handles_labels()
        
        if ax2 is not None and rot_col and rot_col in subset.columns:
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles.extend(handles2)
            labels.extend(labels2)
        
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        

        # 4. Wind Rose / Arrow (Top Right)
        if wind_speed_col and wind_dir_col and wind_speed_col in subset.columns and wind_dir_col in subset.columns:
            valid_wind = self._clean_nans(subset, [wind_speed_col, wind_dir_col])
            if not valid_wind.empty:
                # average true wind speed
                mean_ws = (np.mean(valid_wind[wind_speed_col]))
                
                # Average true wind direction (cartesian mean over sin/cos to avoid 359/1 error)
                wd_rad = np.deg2rad(valid_wind[wind_dir_col])
                mean_wd_rad = np.arctan2(np.mean(np.sin(wd_rad)), np.mean(np.cos(wd_rad)))
                mean_wd = np.rad2deg(mean_wd_rad) % 360
                
                # Calculate relative wind direction to zero_heading
                rel_wd = (mean_wd - zero_heading) % 360
                rel_wd_rad = np.deg2rad(rel_wd)
                
                # Create inset polar axis in the top right corner
                ax_inset = ax.inset_axes([0.85, 0.65, 0.15, 0.35], polar=True)
                ax_inset.set_theta_zero_location("E") # Top is zero_heading
                ax_inset.set_theta_direction(-1)      # Clockwise
                
                ax_inset.set_ylim(0, 1.0)
                ax_inset.set_yticks([])
                ax_inset.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
                ax_inset.set_xticklabels([f'{zero_heading}°', f'{(zero_heading+90)%360}°', f'{(zero_heading-180)%360}°', f'{(zero_heading-90)%360}°'], fontsize=7)
                
                # Draw wind arrow (pointing from where the wind comes towards the center)
                ax_inset.annotate('', xy=(rel_wd_rad - np.pi, 0.2), xytext=(rel_wd_rad, 0.9),
                                  arrowprops=dict(facecolor='black', width=2, headwidth=5, shrink=0.0))
                
                # Write the text
                ax_inset.text(-0.6, 0.5, f"{mean_ws:.1f} m/s\n {mean_wd:.1f}°", transform=ax_inset.transAxes,
                              ha='center', va='center', fontsize=9, fontweight='bold', color='black')
            else:
                print(f"Warning: Only NaN values for wind data in the given window.")

        if invert_y:
            ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def compare_azimuth(
        self,
        trial_list: list,
        zero_point: float = 0.0,
        heading_col: str | None = None,
        zero_heading: float = 0.0,
        rot_col: str | None = None,
        rot_scale: float = 1.0,
        rot_clamp_min: float = None,
        rot_clamp_max: float = None,
        figsize: tuple = (18, 4),
        title: str = "Azimuth Comparison",
        scatter: bool = False,
        clamp_min: float = None,
        clamp_max: float = None,
        invert_y: bool = False,
    ) -> None:
        """
        Vergleiche mehrere Trials nebeneinander in Subplots.
        X-Achsen bleiben konsistent: längere Trials sind auch breiter dargestellt.
        Alle Subplots sind linksbündig aligned und haben synchronisierte Y-Achsen.

        Parameters
        ----------
        trial_list : list – Liste von Dicts mit {"df": df, "start": str, "end": str, "name": str, "columns": list}
        zero_point : float – Ruder-Nullpunkt
        heading_col : str – optional Heading-Spalte
        zero_heading : float – Schiff-Nullpunkt
        rot_col : str – optional Rate of Turn Spalte
        rot_scale : float – Skalierungsfaktor für ROT-Achse
        rot_clamp_min/max : float – Begrenzungen für ROT
        figsize : tuple – Gesamt-Größe der Figure
        title : str – Titel für die ganze Comparison
        clamp_min/max : float – Begrenzungen für Hauptachse
        invert_y : bool – Y-Achse invertieren
        """
        if not trial_list or len(trial_list) == 0:
            print("Error: trial_list ist leer")
            return

        # Schritt 1: Berechne Dauer jedes Trials (in Sekunden)
        durations = []
        for trial in trial_list:
            t0 = self._parse_ts(trial["start"])
            t1 = self._parse_ts(trial["end"])
            duration_secs = (t1 - t0).total_seconds()
            durations.append(duration_secs)

        # Schritt 2: Normalisiere auf relative Breiten (proportional zur Dauer)
        max_duration = max(durations)
        relative_widths = [d / max_duration for d in durations]

        # Schritt 3: Berechne globale Y-Limits über alle Trials und Spalten
        global_ymin = float('inf')
        global_ymax = float('-inf')

        for trial in trial_list:
            subset = self._filter_window(trial["start"], trial["end"])
            if subset.empty:
                continue

            for col in trial["columns"]:
                if col in subset.columns:
                    valid_data = self._clean_nans(subset, col)
                    if not valid_data.empty:
                        vals = self.clamp(valid_data[col], clamp_min, clamp_max)
                        global_ymin = min(global_ymin, vals.min())
                        global_ymax = max(global_ymax, vals.max())

            # Auch Heading-Daten einbeziehen
            if heading_col and heading_col in subset.columns:
                valid_heading = self._clean_nans(subset, heading_col).copy()
                if not valid_heading.empty:
                    valid_heading[heading_col] = self._map_angle_to_180(valid_heading[heading_col], zero_heading)
                    vals = self.clamp(valid_heading[heading_col], clamp_min, clamp_max)
                    global_ymin = min(global_ymin, vals.min())
                    global_ymax = max(global_ymax, vals.max())

        # Mache symmetrisch um 0
        if global_ymin == float('inf') or global_ymax == float('-inf'):
            print("Warning: Keine Daten gefunden")
            return

        global_yabs_max = max(abs(global_ymin), abs(global_ymax))
        global_ymin = -global_yabs_max
        global_ymax = global_yabs_max

        # Schritt 4: Erstelle Subplots mit GridSpec (unterschiedliche Breiten)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, len(trial_list), width_ratios=relative_widths, wspace=0.3)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(trial_list))]

        # Schritt 5: Plotte jeden Trial
        all_handles = []
        all_labels = []

        for idx, (ax, trial) in enumerate(zip(axes, trial_list)):
            subset = self._filter_window(trial["start"], trial["end"])
            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue

            def _plot_series(axis, x_values, y_values, label, *, color=None, linestyle="-", linewidth=1.5):
                if scatter:
                    axis.scatter(x_values, y_values, label=label, s=10, alpha=0.8, color=color)
                else:
                    axis.plot(x_values, y_values, label=label, linestyle=linestyle, linewidth=linewidth, color=color)

            # Plot Azimuth/Rudder Spalten
            for col in trial["columns"]:
                if col in subset.columns:
                    valid_data = self._clean_nans(subset, col).copy()
                    if not valid_data.empty:
                        valid_data[col] = ((valid_data[col] - zero_point + 180) % 360) - 180
                        y = self.clamp(valid_data[col], clamp_min, clamp_max)
                        style = self.COLUMN_STYLE.get(col, {"linestyle": "-", "linewidth": 1.5})
                        _plot_series(ax, valid_data["_dt_local"], y, label=col, **style)

            # Plot Heading Change
            if heading_col and heading_col in subset.columns:
                valid_heading = self._clean_nans(subset, heading_col).copy()
                if not valid_heading.empty:
                    valid_heading[heading_col] = self._map_angle_to_180(valid_heading[heading_col], zero_heading)
                    y = self.clamp(valid_heading[heading_col], clamp_min, clamp_max)
                    style = self.COLUMN_STYLE.get(heading_col, {"linestyle": "--", "linewidth": 2.0})
                    _plot_series(
                        ax,
                        valid_heading["_dt_local"],
                        y,
                        label=f"Heading ({heading_col})",
                        **style
                    )

            # Plot ROT on secondary axis
            ax2 = None
            if rot_col and rot_col in subset.columns:
                valid_rot = self._clean_nans(subset, rot_col).copy()
                if not valid_rot.empty:
                    y_rot = self.clamp(valid_rot[rot_col], rot_clamp_min, rot_clamp_max)
                    ax2 = ax.twinx()
                    style = self.COLUMN_STYLE.get(rot_col, {"color": "black", "linestyle": ":"})
                    _plot_series(ax2, valid_rot["_dt_local"], y_rot, label=f"ROT", **style)
                    
                    # Synchronisiere ROT-Achse (symmetrisch wie Hauptachse)
                    rot_range = max(abs(y_rot.min()), abs(y_rot.max())) if len(y_rot) > 0 else 1
                    if rot_range > 0:
                        scale_factor = (global_yabs_max / rot_range) * rot_scale
                        ax2.set_ylim(-rot_range * scale_factor, rot_range * scale_factor)
                    else:
                        ax2.set_ylim(global_ymin, global_ymax)
                    
                    # ROT-Label nur am rechten Subplot
                    if idx == len(trial_list) - 1:
                        ax2.set_ylabel("Rate of Turn [°/min]", fontsize=9)
                    else:
                        ax2.set_yticklabels([])

            # Setze Y-Limits (alle gleich)
            ax.set_ylim(global_ymin, global_ymax)
            ax.grid(True, which='both', axis='y', alpha=0.7)

            # Labels und Title
            ax.set_xlabel("Time", fontsize=9)
            if idx == 0:
                ax.set_ylabel("Angle [°]", fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.set_title(f"{trial['name']}\n({durations[idx]:.1f}s)", fontsize=10, fontweight='bold')

            # X-Achsen-Formatter
            duration_mins = durations[idx] / 60.0
            if duration_mins <= 3:
                ax.xaxis.set_major_locator(mdates.SecondLocator(bysecond=[0, 30]))
            elif duration_mins <= 5:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            elif duration_mins <= 15:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
            else:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz="Europe/Amsterdam"))

            # Legendenelemente sammeln (nur vom ersten Subplot)
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                all_handles.extend(handles)
                all_labels.extend(labels)
                if ax2 is not None:
                    h2, l2 = ax2.get_legend_handles_labels()
                    all_handles.extend(h2)
                    all_labels.extend(l2)

            if invert_y:
                ax.invert_yaxis()

        # Gemeinsame Legend oben
        if all_handles:
            fig.legend(all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=9)

        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
