import numpy as np

rng = np.random.default_rng()


def sin_data(x):
    new_xs = x / len(x) * 2 * np.pi * 2
    data = np.sin(new_xs)
    return data


def sin_cos_data(x):
    new_xs = x / len(x) * 2 * np.pi * 8
    sin_cos = np.sin(new_xs) + np.cos(new_xs / 2)
    return sin_cos


def linear_data(x):
    linear = x / 1000
    return linear


def normal_noise_per_time_point(noise_factor, this_x, time_point_noise_probability):
    noise = add_normal_noise_to_series(np.zeros(len(this_x)), noise_factor)
    noise_locations = rng.uniform(size=len(this_x)) <= time_point_noise_probability
    noise = noise * noise_locations
    return noise


def exponentially_distributed_noise(this_x, time_point_noise_probability):
    noise = np.zeros(len(this_x))
    if time_point_noise_probability > 0:
        anomaly_locations = get_anomaly_locations(this_x, time_point_noise_probability)
        noise[anomaly_locations] = 1
    return noise


def exponentially_distributed_cluster_noise(this_x, time_point_noise_probability):
    cluster_size = 5
    time_point_noise_probability = time_point_noise_probability / cluster_size
    noise = np.zeros(len(this_x))
    if time_point_noise_probability > 0:
        anomaly_locations = get_anomaly_locations(this_x, time_point_noise_probability)
        final_locations = construct_clustered_anomalies(this_x, anomaly_locations,
                                                        half_cluster=int(np.floor(cluster_size / 2)))
        noise[final_locations] = 1
    return noise


def get_anomaly_locations(x, time_point_noise_probability):
    n_expected_events = int(len(x) * time_point_noise_probability)
    time_to_anomaly = rng.exponential(1 / time_point_noise_probability, size=n_expected_events)
    anomaly_locations = [int(item) for item in np.cumsum(time_to_anomaly) if int(item) < len(x)]
    return anomaly_locations


def construct_clustered_anomalies(x, anomaly_locations, half_cluster):
    final_locations = []
    for location in anomaly_locations:
        append_locations_in_cluster(final_locations, half_cluster, location)
    final_locations = np.array(final_locations, dtype=int)
    final_locations = final_locations[(final_locations >= 0) & (final_locations < len(x))]
    return final_locations


def append_locations_in_cluster(final_locations, half_cluster, location):
    for i in range(-half_cluster, half_cluster, 1):
        final_locations.append(int(location) + i)


def add_normal_noise_to_series(series, noise_factor):
    len_series = len(series)
    return series + rng.normal(scale=noise_factor, size=len_series)
