import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter, MerweScaledSigmaPoints, UnscentedKalmanFilter
from sklearn.metrics import mean_squared_error

########################################### Utils ############################################


KNOTS_TO_MS: float = 0.51444
DEG_TO_RAD: float = np.pi / 180.0
RAD_TO_DEG: float = 180.0 / np.pi
EARTH_RADIUS: float = 6378137.0


def geo_to_cartesian(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat_rad = lat * DEG_TO_RAD
    lon_rad = lon * DEG_TO_RAD
    lat0_rad = lat0 * DEG_TO_RAD
    lon0_rad = lon0 * DEG_TO_RAD

    x = EARTH_RADIUS * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = EARTH_RADIUS * (lat_rad - lat0_rad)

    return x, y


def cartesian_to_geo(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat0_rad = lat0 * DEG_TO_RAD
    lon0_rad = lon0 * DEG_TO_RAD

    lat_rad = lat0_rad + y / EARTH_RADIUS
    lon_rad = lon0_rad + x / (EARTH_RADIUS * np.cos(lat0_rad))

    lat = lat_rad * RAD_TO_DEG
    lon = lon_rad * RAD_TO_DEG

    return lat, lon


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = lat1 * DEG_TO_RAD
    lon1_rad = lon1 * DEG_TO_RAD
    lat2_rad = lat2 * DEG_TO_RAD
    lon2_rad = lon2 * DEG_TO_RAD

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS * c


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


########################################### Kalman Filter ############################################


def init(
    init_state: np.ndarray,
    dt: float = 1.0,  # sec
    model_type: str = "ca",
    params: dict | None = None,
) -> KalmanFilter:
    """
    Initialize a Kalman filter for ship trajectory prediction

    :param init_state: Initial state vector
    :param dt: Time step in seconds
    :param model_type: Type of Kalman filter model ('ca', 'ukf', 'ckf')
    :param params: Optional dictionary with tuned parameters
    """
    if params is None:
        params = {}

    process_noise = params.get("process_noise", 0.01)
    measurement_noise = params.get("measurement_noise", 0.05)

    if model_type == "ca":  # constant acceleration model
        # state: [x, y, vx, vy, ax, ay]
        kf = KalmanFilter(dim_x=6, dim_z=2)

        kf.F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        )

        # measurement function: only observe position)
        kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        kf.x = np.zeros((6, 1))
        kf.x[:2, 0] = init_state[:2]

        if len(init_state) >= 4:
            kf.x[2:4, 0] = init_state[2:4]
        if len(init_state) >= 6:
            kf.x[4:6, 0] = init_state[4:6]

        q = process_noise
        # NOTE: Q_discrete_white_noise works only with 2<=dim_x<=4 therefore manual
        kf.Q = np.zeros((6, 6))
        # position block (x, y)
        kf.Q[0:2, 0:2] = np.eye(2) * q * dt**5 / 20
        # velocity block (vx, vy)
        kf.Q[2:4, 2:4] = np.eye(2) * q * dt**3 / 3
        # aceleration block (ax, ay)
        kf.Q[4:6, 4:6] = np.eye(2) * q * dt
        # cross-correlation blocks
        kf.Q[0:2, 2:4] = kf.Q[2:4, 0:2] = np.eye(2) * q * dt**4 / 8
        kf.Q[0:2, 4:6] = kf.Q[4:6, 0:2] = np.eye(2) * q * dt**3 / 6
        kf.Q[2:4, 4:6] = kf.Q[4:6, 2:4] = np.eye(2) * q * dt**2 / 2

        # measurement noise with optionally tuned parameters
        kf.R = np.eye(2) * measurement_noise

        # initial cov
        kf.P = np.eye(6) * 100
    # complete kinematic filter / extended kalman filter -- same state space, different predict step
    elif model_type in ("ckf", "ekf"):
        heading_rad = init_state[2] * DEG_TO_RAD
        angular_rate_rad = init_state[3] * DEG_TO_RAD
        transversal_speed_ms = init_state[4] * KNOTS_TO_MS
        longitudal_speed_ms = init_state[5] * KNOTS_TO_MS

        # state: [x, y, heading, angular_rate, transversal_speed, longitudal_speed]
        kf = KalmanFilter(dim_x=6, dim_z=6)

        # NOTE: calculations are deliberately stolen from the Internet
        kf.F = np.array(
            [
                [
                    1,
                    0,
                    dt * longitudal_speed_ms * np.cos(heading_rad)
                    - dt * transversal_speed_ms * np.sin(heading_rad),
                    0,
                    dt * np.cos(heading_rad),
                    dt * np.sin(heading_rad),
                ],
                [
                    0,
                    1,
                    -dt * longitudal_speed_ms * np.sin(heading_rad)
                    - dt * transversal_speed_ms * np.cos(heading_rad),
                    0,
                    -dt * np.sin(heading_rad),
                    dt * np.cos(heading_rad),
                ],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        )

        # observe all state variables
        kf.H = np.eye(6)
        kf.x = np.zeros((6, 1))
        kf.x[0, 0] = init_state[0]  # x position
        kf.x[1, 0] = init_state[1]  # y position
        kf.x[2, 0] = heading_rad  # heading in radians
        kf.x[3, 0] = angular_rate_rad  # angular rate in radians
        kf.x[4, 0] = transversal_speed_ms  # transversal speed in m/s
        kf.x[5, 0] = longitudal_speed_ms  # longitudal speed in m/s

        # default params are deliberately stolen from the Internet
        process_noise_pos = params.get("process_noise_pos", process_noise)
        process_noise_heading = params.get("process_noise_heading", 0.005)
        process_noise_angular = params.get("process_noise_angular", 0.01)
        process_noise_speed = params.get("process_noise_speed", 0.05)

        kf.Q = np.diag(
            [
                process_noise_pos,  # x position
                process_noise_pos,  # y position
                process_noise_heading,  # heading
                process_noise_angular,  # angular rate
                process_noise_speed,  # transversal speed
                process_noise_speed,  # longitudal speed
            ],
        )

        measurement_noise_pos = params.get("measurement_noise_pos", measurement_noise)
        measurement_noise_heading = params.get("measurement_noise_heading", 0.02)
        measurement_noise_angular = params.get("measurement_noise_angular", 0.01)
        measurement_noise_speed = params.get("measurement_noise_speed", 0.05)

        kf.R = np.diag(
            [
                measurement_noise_pos,  # x position
                measurement_noise_pos,  # y position
                measurement_noise_heading,  # heading
                measurement_noise_angular,  # angular rate
                measurement_noise_speed,  # transversal speed
                measurement_noise_speed,  # longitudal speed
            ],
        )

        kf.P = np.diag(
            [
                100.0,  # x position
                100.0,  # y position
                np.pi / 4,  # heading
                0.1,  # angular rate
                1.0,  # transversal speed
                1.0,  # longitudal speed
            ],
        )
    # unscented kalman filter: same 6D kinematic state as ckf/ekf, sigma points propagate exact f(x)
    elif model_type == "ukf":
        heading_rad = init_state[2] * DEG_TO_RAD
        angular_rate_rad = init_state[3] * DEG_TO_RAD
        transversal_speed_ms = init_state[4] * KNOTS_TO_MS
        longitudal_speed_ms = init_state[5] * KNOTS_TO_MS

        alpha = params.get("alpha", 0.9)
        beta = params.get("beta", 2.0)
        kappa = params.get("kappa", 0.0)

        points = MerweScaledSigmaPoints(n=6, alpha=alpha, beta=beta, kappa=kappa)
        kf = UnscentedKalmanFilter(
            dim_x=6,
            dim_z=6,
            dt=dt,
            fx=_ekf_f,
            hx=lambda x: x,
            points=points,
            x_mean_fn=_ukf_state_mean,
            z_mean_fn=_ukf_state_mean,
            residual_x=_ukf_residual,
            residual_z=_ukf_residual,
        )

        kf.x = np.zeros(6)
        kf.x[0] = init_state[0]
        kf.x[1] = init_state[1]
        kf.x[2] = heading_rad
        kf.x[3] = angular_rate_rad
        kf.x[4] = transversal_speed_ms
        kf.x[5] = longitudal_speed_ms

        process_noise_pos = params.get("process_noise_pos", process_noise)
        process_noise_heading = params.get("process_noise_heading", 0.005)
        process_noise_angular = params.get("process_noise_angular", 0.01)
        process_noise_speed = params.get("process_noise_speed", 0.05)

        kf.Q = np.diag(
            [
                process_noise_pos,
                process_noise_pos,
                process_noise_heading,
                process_noise_angular,
                process_noise_speed,
                process_noise_speed,
            ],
        )

        measurement_noise_pos = params.get("measurement_noise_pos", measurement_noise)
        measurement_noise_heading = params.get("measurement_noise_heading", 0.02)
        measurement_noise_angular = params.get("measurement_noise_angular", 0.01)
        measurement_noise_speed = params.get("measurement_noise_speed", 0.05)

        kf.R = np.diag(
            [
                measurement_noise_pos,
                measurement_noise_pos,
                measurement_noise_heading,
                measurement_noise_angular,
                measurement_noise_speed,
                measurement_noise_speed,
            ],
        )

        kf.P = np.diag([100.0, 100.0, np.pi / 4, 0.1, 1.0, 1.0])

    else:
        raise ValueError(f"unsupported model: {model_type}")
    return kf


def estimate_initial_velocity(positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Estimate initial velocity from positions historey

    :param positions: Positions with shape [seq_len, 2]
    :param dt: Time step between measurements
    :return velocity: Estimated velocity [vx, vy]
    """
    if len(positions) < 3:
        if len(positions) >= 2:
            velocity = (positions[-1] - positions[-2]) / dt
        else:
            velocity = np.zeros(2)
    else:
        # lin polynomial regression
        t = np.arange(len(positions)) * dt
        vx = np.polyfit(t, positions[:, 0], 1)[0]
        vy = np.polyfit(t, positions[:, 1], 1)[0]
        velocity = np.array([vx, vy])
    return velocity


def estimate_initial_acceleration(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Estimate initial acceleration from position and velocity history

    :param positions: Position history with shape [seq_len, 2]
    :param velocities: Velocity history with shape [seq_len-1, 2]
    :param dt: Time step between measurements
    :return acceleration: Estimated acceleration [ax, ay]
    """
    if len(positions) < 4:
        if len(velocities) >= 2:
            acceleration = (velocities[-1] - velocities[-2]) / dt
        else:
            # assume zero acceleration
            acceleration = np.zeros(2)
    else:
        # quadratic polynomial regression
        t = np.arange(len(positions)) * dt
        ax = np.polyfit(t, positions[:, 0], 2)[0] * 2  # a = 2nd coef * 2
        ay = np.polyfit(t, positions[:, 1], 2)[0] * 2
        acceleration = np.array([ax, ay])

    return acceleration


def _ekf_f(x: np.ndarray, dt: float) -> np.ndarray:
    """Nonlinear ship kinematics state transition.
    Accepts 1D (UKF sigma points) and (6,1) column (KF) arrays"""
    h, omega, v_trans, v_long = x[2], x[3], x[4], x[5]
    x_new = x.copy()
    x_new[0] += dt * (v_long * np.sin(h) + v_trans * np.cos(h))
    x_new[1] += dt * (v_long * np.cos(h) - v_trans * np.sin(h))
    x_new[2] = wrap_angle(x_new[2] + omega * dt)
    return x_new


def _ukf_state_mean(sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted sigma-point mean with circular averaging for the heading (index 2)"""
    mean = np.zeros(6)
    for i in (0, 1, 3, 4, 5):
        mean[i] = np.dot(weights, sigmas[:, i])
    sin_sum = np.dot(weights, np.sin(sigmas[:, 2]))
    cos_sum = np.dot(weights, np.cos(sigmas[:, 2]))
    mean[2] = np.arctan2(sin_sum, cos_sum)
    return mean


def _ukf_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """State/measurement residual with the heading (index 2)"""
    y = np.subtract(a, b)
    y[2] = wrap_angle(y[2])
    return y


def _ekf_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    h, v_trans, v_long = float(np.squeeze(x[2])), float(np.squeeze(x[4])), float(np.squeeze(x[5]))
    F = np.eye(6)
    F[0, 2] = dt * (v_long * np.cos(h) - v_trans * np.sin(h))
    F[0, 4] = dt * np.cos(h)
    F[0, 5] = dt * np.sin(h)
    F[1, 2] = dt * (-v_long * np.sin(h) - v_trans * np.cos(h))
    F[1, 4] = -dt * np.sin(h)
    F[1, 5] = dt * np.cos(h)
    F[2, 3] = dt
    return F


def predict(
    sequence: np.ndarray,
    pred_len: int,
    dt: float = 1.0,
    model_type: str = "ckf",
    ref_coords: tuple[float, float] | None = None,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict based on input sequence

    :param sequence: Input sequence with shape [seq_len, features]
    :param pred_len: Number of steps to predict
    :param dt: Time step between measurements (in seconds)
    :param model_type: Type of motion model to use ('ca', 'ukf', or 'ckf')
    :param ref_coords: Optional Reference coordinates (lat0, lon0) for local coordinate conversion
    :param params: Optional dictionary with tuned parameters for the kf
    :return predictions: Kalman filter predictions with shape [pred_len, features]
    :return uncertainties: Uncertainties in the predictions, i.e. std in lat/lon space
    """
    feature_dim = sequence.shape[1]
    if ref_coords is None:
        lat0 = np.mean(sequence[:, 0])
        lon0 = np.mean(sequence[:, 1])
    else:
        lat0, lon0 = ref_coords

    cartesian = np.zeros((sequence.shape[0], 2))
    for i in range(sequence.shape[0]):
        cartesian[i, 0], cartesian[i, 1] = geo_to_cartesian(
            sequence[i, 0],
            sequence[i, 1],
            lat0,
            lon0,
        )

    if model_type in ("ckf", "ekf", "ukf"):
        # latitude, longitude, heading_angle, heading_angular_rate, transversal_speed, longitudal_speed
        x, y = cartesian[-1]
        heading_angle = sequence[-1, 2]
        heading_angular_rate = sequence[-1, 3]
        transversal_speed = sequence[-1, 4]
        longitudal_speed = sequence[-1, 5]

        init_state = np.array(
            [
                x,
                y,
                heading_angle,
                heading_angular_rate,
                transversal_speed,
                longitudal_speed,
            ],
        )

        kf = init(init_state, dt, model_type, params=params)

        for i in range(len(sequence)):
            # "training" the filter with measurement data
            x, y = cartesian[i]
            heading_angle = sequence[i, 2]
            heading_angular_rate = sequence[i, 3]
            transversal_speed = sequence[i, 4]
            longitudal_speed = sequence[i, 5]

            measurement_vector = np.array(
                [
                    x,
                    y,
                    heading_angle * DEG_TO_RAD,
                    heading_angular_rate * DEG_TO_RAD,
                    transversal_speed * KNOTS_TO_MS,
                    longitudal_speed * KNOTS_TO_MS,
                ],
            )

            # NOTE: align the measured heading to the current estimate so the linear innovation z- Hx stays ok
            # across the 0/2*pi discontinuity,filterpy's linear KalmanFilter has no custom-residual hook; the
            # UKF handles this via _ukf_residual instead
            state_heading = kf.x[2] if model_type == "ukf" else kf.x[2, 0]
            measurement_vector[2] = state_heading + wrap_angle(
                measurement_vector[2] - state_heading,
            )

            kf.update(measurement_vector.reshape(-1))

            if i < len(sequence) - 1:
                if model_type == "ckf":
                    heading_rad = kf.x[2, 0]
                    trans_speed_ms = kf.x[4, 0]
                    long_speed_ms = kf.x[5, 0]

                    kf.F[0, 2] = dt * long_speed_ms * np.cos(
                        heading_rad,
                    ) - dt * trans_speed_ms * np.sin(heading_rad)
                    kf.F[0, 4] = dt * np.cos(heading_rad)
                    kf.F[0, 5] = dt * np.sin(heading_rad)
                    kf.F[1, 2] = -dt * long_speed_ms * np.sin(
                        heading_rad,
                    ) - dt * trans_speed_ms * np.cos(heading_rad)
                    kf.F[1, 4] = -dt * np.sin(heading_rad)
                    kf.F[1, 5] = dt * np.cos(heading_rad)

                    kf.predict()
                    kf.x[2, 0] = wrap_angle(kf.x[2, 0])
                else:  # ekf: exact nonlinear state, Jacobian for covariance
                    F_jac = _ekf_jacobian(kf.x, dt)
                    kf.x = _ekf_f(kf.x, dt)
                    kf.P = F_jac @ kf.P @ F_jac.T + kf.Q
                # ukf: sigma points propagate exact f(x) internally via kf.predict()
                # (no explicit branch needed — filterpy UKF handles it)

        predictions = np.zeros((pred_len, feature_dim))
        uncertainties = np.zeros((pred_len, 2))
        m_to_lat = 1.0 / (EARTH_RADIUS * DEG_TO_RAD)
        m_to_lon = 1.0 / (EARTH_RADIUS * np.cos(lat0 * DEG_TO_RAD) * DEG_TO_RAD)
        for i in range(pred_len):
            if model_type == "ckf":
                heading_rad = kf.x[2, 0]
                trans_speed_ms = kf.x[4, 0]
                long_speed_ms = kf.x[5, 0]

                kf.F[0, 2] = dt * long_speed_ms * np.cos(
                    heading_rad,
                ) - dt * trans_speed_ms * np.sin(heading_rad)
                kf.F[0, 4] = dt * np.cos(heading_rad)
                kf.F[0, 5] = dt * np.sin(heading_rad)
                kf.F[1, 2] = -dt * long_speed_ms * np.sin(
                    heading_rad,
                ) - dt * trans_speed_ms * np.cos(heading_rad)
                kf.F[1, 4] = -dt * np.sin(heading_rad)
                kf.F[1, 5] = dt * np.cos(heading_rad)

                kf.predict()
                kf.x[2, 0] = wrap_angle(kf.x[2, 0])
            elif model_type == "ekf":
                F_jac = _ekf_jacobian(kf.x, dt)
                kf.x = _ekf_f(kf.x, dt)
                kf.P = F_jac @ kf.P @ F_jac.T + kf.Q
            else:  # ukf
                kf.predict()

            # UKF state is 1D; KF state is (6,1) column vector
            x_flat = kf.x if model_type == "ukf" else kf.x[:, 0]
            lat, lon = cartesian_to_geo(x_flat[0], x_flat[1], lat0, lon0)

            predictions[i, 0] = lat
            predictions[i, 1] = lon
            predictions[i, 2] = (x_flat[2] * RAD_TO_DEG) % 360.0
            predictions[i, 3] = x_flat[3] * RAD_TO_DEG
            predictions[i, 4] = x_flat[4] / KNOTS_TO_MS
            predictions[i, 5] = x_flat[5] / KNOTS_TO_MS

            uncertainties[i, 0] = np.sqrt(kf.P[0, 0]) * m_to_lat
            uncertainties[i, 1] = np.sqrt(kf.P[1, 1]) * m_to_lon
    else:  # ca
        positions = cartesian
        velocities = np.zeros((len(positions) - 1, 2))
        for i in range(len(positions) - 1):
            velocities[i] = (positions[i + 1] - positions[i]) / dt

        velocity = estimate_initial_velocity(positions, dt)
        accel = estimate_initial_acceleration(positions, velocities, dt)
        init_state = np.concatenate([positions[-1], velocity, accel])
        kf = init(init_state, dt, model_type, params)

        for i in range(len(positions)):
            kf.update(positions[i].reshape(-1))
            if i < len(positions) - 1:
                kf.predict()

        predictions = np.zeros((pred_len, feature_dim))
        uncertainties = np.zeros((pred_len, 2))
        m_to_lat = 1.0 / (EARTH_RADIUS * DEG_TO_RAD)
        m_to_lon = 1.0 / (EARTH_RADIUS * np.cos(lat0 * DEG_TO_RAD) * DEG_TO_RAD)
        for i in range(pred_len):
            kf.predict()
            local_x, local_y = kf.x[:2, 0]
            lat, lon = cartesian_to_geo(local_x, local_y, lat0, lon0)
            predictions[i, 0] = lat
            predictions[i, 1] = lon
            if feature_dim > 2:
                predictions[i, 2:] = sequence[-1, 2:]
            uncertainties[i, 0] = np.sqrt(kf.P[0, 0]) * m_to_lat
            uncertainties[i, 1] = np.sqrt(kf.P[1, 1]) * m_to_lon

    return predictions, uncertainties


def tune_kalman_params(
    train_data_path: str,
    model_type: str,
    seq_len: int = 8,
    pred_len: int = 16,
    dt: float = 1.0,
    ref_coords: tuple[float, float] | None = None,
) -> dict:
    """
    Tune Kalman filter parameters using training data

    :param train_data_path: Path to training data CSV file
    :param model_type: Type of Kalman filter model ('ca', 'ukf', 'ckf')
    :param seq_len: Length of input sequence to use for tuning
    :param pred_len: Length of prediction to evaluate
    :param dt: Time step between measurements (in seconds)
    :param ref_coords: Optional reference coordinates (lat0, lon0) for local coordinate conversion
    :return: Dict param
    """
    df = pd.read_csv(train_data_path)

    feature_cols = list(df.columns)
    if "timestamp" in feature_cols:
        feature_cols.remove("timestamp")

    if model_type in ("ckf", "ekf", "ukf"):
        param_grid = {
            "process_noise_pos": [0.001, 0.01, 0.1],
            "process_noise_heading": [0.001, 0.005, 0.01],
            "measurement_noise_pos": [0.01, 0.05, 0.1],
        }
        # generate all combinations (simplified approach)
        param_combinations = [
            {
                "process_noise_pos": pnp,
                "process_noise_heading": pnh,
                "measurement_noise_pos": mnp,
            }
            for pnp in param_grid["process_noise_pos"]
            for pnh in param_grid["process_noise_heading"]
            for mnp in param_grid["measurement_noise_pos"]
        ]
    else:
        # simpler grid for CA and UKF models
        param_grid = {
            "process_noise": [0.001, 0.01, 0.1],
            "measurement_noise": [0.01, 0.05, 0.1],
        }
        param_combinations = [
            {"process_noise": pn, "measurement_noise": mn}
            for pn in param_grid["process_noise"]
            for mn in param_grid["measurement_noise"]
        ]

    best_params = None
    best_error = float("inf")

    if ref_coords is None:
        lat0 = df[feature_cols[0]].mean()
        lon0 = df[feature_cols[1]].mean()
        ref_coords = (lat0, lon0)

    # grid search
    for params in param_combinations:
        total_error = 0
        samples = 0

        for _ in range(5):  # eval 5 random samples
            start_pos = random.randint(0, len(df) - seq_len - pred_len)

            input_sequence = df.iloc[start_pos : start_pos + seq_len][feature_cols].values
            true_future = df.iloc[start_pos + seq_len : start_pos + seq_len + pred_len][feature_cols].values

            predictions, _ = predict(
                sequence=input_sequence,
                pred_len=pred_len,
                dt=dt,
                model_type=model_type,
                ref_coords=ref_coords,
                params=params,
            )

            # calculate error for latitude and longitude
            mse = mean_squared_error(true_future[:, :2], predictions[:, :2])
            total_error += mse
            samples += 1

        avg_error = total_error / max(1, samples)
        if avg_error < best_error:
            best_error = avg_error
            best_params = params

    return best_params


def save_params(params: dict, filename: str) -> None:
    p = Path(filename)
    p.resolve().parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(params, f, indent=4)


def load_params(filename: str) -> dict:
    p = Path(filename)
    if not p.exists():
        raise FileNotFoundError(f"file {filename} not found")
    with p.open() as f:
        return json.load(f)


def plot(
    sequence: np.ndarray,
    true_future: np.ndarray,
    kalman_predictions: np.ndarray,
    kalman_uncertainties: np.ndarray | None = None,
    model_type: str = "ckf",
    output_dir: str | None = None,
    start_pos: int = 0,
):
    """
    Plot KF predictions against true future values

    :param sequence: Input sequence data [seq_len, features]
    :param true_future: Ground truth future values [pred_len, features]
    :param kalman_predictions: Kalman filter predictions [pred_len, features]
    :param kalman_uncertainties: Optional uncertainties in Kalman predictions [pred_len, 2]
    :param model_type: Kalman filter model type
    :param output_dir: Directory to save plot
    :param start_pos: Starting position in dataset (for naming)
    """
    _, axes = plt.subplots(nrows=2, figsize=(14, 10), sharex=True)
    feature_names = ["Latitude", "Longitude"]

    seq_len = sequence.shape[0]
    pred_len = true_future.shape[0]
    x_seq = np.arange(seq_len)
    x_future = np.arange(seq_len, seq_len + pred_len)

    for i in range(2):  # only plot latitude and longitude
        ax = axes[i]

        ax.plot(x_seq, sequence[:, i], "b-", linewidth=2, label="Input Sequence")

        ax.plot(
            x_future,
            true_future[:, i],
            "g-",
            marker="o",
            linewidth=2,
            markersize=6,
            label="True Future",
        )

        ax.plot(
            x_future,
            kalman_predictions[:, i],
            "m-",
            marker="x",
            linewidth=2,
            markersize=7,
            label=f"Kalman Filter ({model_type.upper()})",
        )

        if kalman_uncertainties is not None:
            ax.fill_between(
                x_future,
                kalman_predictions[:, i] - 2 * kalman_uncertainties[:, i],  # 2 sigma; 95% confidence
                kalman_predictions[:, i] + 2 * kalman_uncertainties[:, i],
                color="m",
                alpha=0.2,
                label="95% Confidence",
            )

        ax.set_title(f"{feature_names[i]}")
        ax.legend(loc="best")
        ax.grid(True)

    plt.xlabel("Timesteps")
    plt.suptitle(f"Trajectory Prediction using {model_type.upper()} Kalman Filter")
    plt.tight_layout()

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = f"kalman_{model_type}_{start_pos}.png"
        if (out_dir / output_file).exists():
            counter = 2
            output_file = f"kalman_{model_type}_{start_pos}_{counter}.png"
            while (out_dir / output_file).exists():
                counter += 1
                output_file = f"kalman_{model_type}_{start_pos}_{counter}.png"
        out_path = out_dir / output_file
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")


def compute_metrics(true_values: np.ndarray, predictions: np.ndarray) -> dict:
    true_pos = true_values[:, :2]
    pred_pos = predictions[:, :2]

    mse = mean_squared_error(true_pos, pred_pos)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_pos - pred_pos))

    # haversine distance for metric
    distances = [
        haversine_distance(true_pos[i, 0], true_pos[i, 1], pred_pos[i, 0], pred_pos[i, 1])
        for i in range(len(true_pos))
    ]

    mean_distance = np.mean(distances)
    max_distance = np.max(distances)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mean_distance_meters": mean_distance,
        "max_distance_meters": max_distance,
    }


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-d", "--test_data", type=str, required=True, help="test data file")
    parser.add_argument("-t", "--train_data", type=str, default=None, help="training data for tuning")
    parser.add_argument("-m", "--model", type=str, default="ckf", choices=["ca", "ukf", "ckf", "ekf"],
        help="kf model type: ca (constant acceleration), ukf (unscented kalman filter), ckf (complete kinematic filter), ekf (extended kalman filter)")  # noqa: E501
    parser.add_argument("-o", "--plots_dir", type=str, default="./plots_kalman",help="plots dir")
    parser.add_argument("-sp", "--start_pos", type=int, default=-1, help="starting position for a seq (-1 for random)")  # noqa: E501
    parser.add_argument("-seq", "--seq_len", type=int, default=8, help="sequence length")
    parser.add_argument("-n", "--num_predictions", type=int, default=16, help="prediction length")
    parser.add_argument("--tune", action="store_true", help="tune kf?")
    parser.add_argument("--save_params", type=str, default=None, help="path to save tuned parameters")
    parser.add_argument("--load_params", type=str, default=None, help="path to load tuned parameters")
    parser.add_argument("--dt", type=float, default=1.0, help="dt bw measurements")
    parser.add_argument("--ref_lat", type=float, default=54.36778,help="Ref lat (appr. Kieler Förde center)")
    parser.add_argument("--ref_lon", type=float,default=10.17306, help="Ref lon (appr. Kieler Förde center)")
    parser.add_argument("--seed", type=int, default=42)
    # fmt: on

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    ref_coords = None
    if args.ref_lat is not None and args.ref_lon is not None:
        ref_coords = (args.ref_lat, args.ref_lon)

    params = None
    if args.load_params:
        try:
            params = load_params(args.load_params)
        except Exception as e:
            print(f"Warning: Failed to load parameters: {e}")

    if args.tune and args.train_data:
        print(f"Tuning Kalman filter parameters for {args.model} model...")
        params = tune_kalman_params(
            train_data_path=args.train_data,
            model_type=args.model,
            dt=args.dt,
            ref_coords=ref_coords,
        )

        if args.save_params:
            save_params(params, args.save_params)

    print(f"Loading test data from {args.test_data}...")
    df = pd.read_csv(args.test_data)

    feature_cols = list(df.columns)
    if "timestamp" in feature_cols:
        feature_cols.remove("timestamp")

    start_pos = args.start_pos
    max_start_pos = len(df) - args.seq_len - args.num_predictions
    if start_pos < 0 or start_pos > max_start_pos:
        start_pos = random.randint(0, max_start_pos)
        print(
            f"Selected random sequence starting at position {start_pos} out of {max_start_pos}",
        )

    input_sequence = df.iloc[start_pos : start_pos + args.seq_len][feature_cols].values
    true_future = df.iloc[start_pos + args.seq_len : start_pos + args.seq_len + args.num_predictions][
        feature_cols
    ].values

    if ref_coords is None:
        lat0 = df[feature_cols[0]].mean()
        lon0 = df[feature_cols[1]].mean()
        ref_coords = (lat0, lon0)
        print(f"Using reference coordinates: lat={lat0:.6f}, lon={lon0:.6f}")

    print(f"Generating {args.model.upper()} Kalman filter predictions...")
    kalman_predictions, kalman_uncertainties = predict(
        sequence=input_sequence,
        pred_len=args.num_predictions,
        dt=args.dt,
        model_type=args.model,
        ref_coords=ref_coords,
        params=params,
    )

    metrics = compute_metrics(true_future, kalman_predictions)
    print("Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")

    plot(
        sequence=input_sequence,
        true_future=true_future,
        kalman_predictions=kalman_predictions,
        kalman_uncertainties=kalman_uncertainties,
        model_type=args.model,
        output_dir=args.plots_dir,
        start_pos=start_pos,
    )


if __name__ == "__main__":
    main()
