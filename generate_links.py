import pandas as pd
import numpy as np
import time
from pathlib import Path

def generate_simple_baltimore(mode = 'test'):

    start_time = 360
    end_time = 720
    chunk_size = 100

    if mode == 'test':
        n_links = 100
        end_time = 420
        output_file = 'Each_iteration_test.csv'

    else:
        n_links = np.inf
        output_file = 'Each_iteration.csv'

        response = input('Proceed? (yes/no): ')
        if response.lower() != 'yes':
            print('Cancelled.')

    print('-' * 65)
    print()

    links = pd.read_excel('Baltimore_Bridge_net.xlsx', sheet_name = 0, header = None)

    links_clean = pd.DataFrame({
        'from_node_id': links.iloc[:, 0],
        'to_node_id': links.iloc[:, 1],
        'length': links.iloc[:, 3],
        'travel_time': links.iloc[:, 4]
    })

    links_clean['speed'] = (links_clean['length'] / links_clean['travel_time']) * 60

    links_clean['speed'] = links_clean['speed'].replace([np.inf, -np.inf], 0).fillna(0)

    if n_links < len(links_clean):
        links_clean = links_clean.iloc[:int(n_links)].copy()

    time_intervals = range(start_time, end_time + 1)
    n_intervals = len(time_intervals)

    total_links = len(links_clean)
    total_rows = total_links * n_intervals

    headers = ['from_node_id', 'to_node_id', 'link_id_from_to', 'day_no', 'timestamp_in_min', 'travel_time_in_min', 'delay_in_min', 'link_in_volume_number_of_veh', 'link_out_volume_number_of_veh', 'link_volume_in_veh_per_hour_per_lane', 'link_volume_in_veh_per_hour_for_all_lanes', 'density_in_veh_per_hour_per_distance_per_lane', 'speed', 'queue_length_percentage', 'number_of_queued_vehicles', 'cumulative_arrival_count', 'cumulative_departure_count', 'total_energy', 'total_CO2', 'total_NOX', 'total_CO', 'total_HC']

    f = open(output_file, 'w')

    f.write(",".join(headers) + '\n')

    n_chunks = int(np.ceil(total_links / chunk_size))
    rows_written = 0

    start_clock = time.time()

    for chunk_idx in range(n_chunks):

        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_links)

        chunk_links = links_clean.iloc[start_idx:end_idx]

        for _, row in chunk_links.iterrows():

            from_node = row['from_node_id']
            to_node = row['to_node_id']

            link_id = f"{int(from_node)}->{int(to_node)}"

            travel_time = row['travel_time']
            speed = row['speed']

            for timestamp in time_intervals:
                f.write(
                    f"{int(from_node)},"
                    f"{int(to_node)},"
                    f"{link_id},"
                    f"1,"
                    f"{timestamp},"
                    f"{travel_time:.2f},"
                    f"0,0,0,0,0,0,"
                    f"{speed:.2f},"
                    f"0,0,0,0,0,0,0,0,0\n"
                )

                rows_written += 1

        if ((chunk_idx + 1) % 10 == 0) or ((chunk_idx + 1) == n_chunks):
            elapsed = time.time - start_clock

            pct_done = 100 * rows_written / total_rows

            if rows_written > 0:
                est_total_time = (
                    elapsed / (rows_written / total_rows)
                )

                est_remaining = (
                    est_total_time - elapsed
                ) / 60.0

            else:
                est_remaining = 0

            print(
                f"Chunk {chunk_idx + 1}/{n_chunks} - "
                f"{pct_done:.1f}% done, "
                f"{est_remaining:.1f} min remaining"
            )

        del chunk_links

    f.close()

    elapsed_total = time.time() - start_clock

    print(f'Finished with total time {elapsed_total}')

def generate_simple_siouxfalls(mode = 'test'):
    start_time = 360
    end_time = 720
    chunk_size = 100

    if mode == 'test':
        n_links = 100
        end_time = 420
        output_file = 'Each_iteration_test.csv'

    else:
        n_links = np.inf
        output_file = 'Each_iteration.csv'

        print('This is not test mode')
        response = input('Proceed? (yes/no): ')
        if response.lower() != 'yes':
            print('Cancelled.')

    print('-' * 65)
    print()

    links = pd.read_excel('SiouxFalls_network.xlsx', skiprows = 7)

    links = links[links['~'].notna()]

    links_clean = pd.DataFrame({
        'from_node_id': links['Init Node'],
        'to_node_id': links.iloc[:, 1],
        'length': links.iloc[:, 3],
        'travel_time': links.iloc[:, 4]
    })

    links_clean['speed'] = (links_clean['length'] / links_clean['travel_time']) * 60

    links_clean['speed'] = links_clean['speed'].replace([np.inf, -np.inf], 0).fillna(0)

    if n_links < len(links_clean):
        links_clean = links_clean.iloc[:int(n_links)].copy()

    time_intervals = range(start_time, end_time + 1)
    n_intervals = len(time_intervals)

    total_links = len(links_clean)
    total_rows = total_links * n_intervals

    headers = ['from_node_id', 'to_node_id', 'link_id_from_to', 'day_no', 'timestamp_in_min', 'travel_time_in_min', 'delay_in_min', 'link_in_volume_number_of_veh', 'link_out_volume_number_of_veh', 'link_volume_in_veh_per_hour_per_lane', 'link_volume_in_veh_per_hour_for_all_lanes', 'density_in_veh_per_hour_per_distance_per_lane', 'speed', 'queue_length_percentage', 'number_of_queued_vehicles', 'cumulative_arrival_count', 'cumulative_departure_count', 'total_energy', 'total_CO2', 'total_NOX', 'total_CO', 'total_HC']

    f = open(output_file, 'w')

    f.write(",".join(headers) + '\n')

    n_chunks = int(np.ceil(total_links / chunk_size))
    rows_written = 0

    start_clock = time.time()

    for chunk_idx in range(n_chunks):

        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_links)

        chunk_links = links_clean.iloc[start_idx:end_idx]

        for _, row in chunk_links.iterrows():

            from_node = row['from_node_id']
            to_node = row['to_node_id']

            link_id = f"{int(from_node)}->{int(to_node)}"

            travel_time = row['travel_time']
            speed = row['speed']

            for timestamp in time_intervals:
                f.write(
                    f"{int(from_node)},"
                    f"{int(to_node)},"
                    f"{link_id},"
                    f"1,"
                    f"{timestamp},"
                    f"{travel_time:.2f},"
                    f"0,0,0,0,0,0,"
                    f"{speed:.2f},"
                    f"0,0,0,0,0,0,0,0,0\n"
                )

                rows_written += 1

        if ((chunk_idx + 1) % 10 == 0) or ((chunk_idx + 1) == n_chunks):
            elapsed = time.time - start_clock

            pct_done = 100 * rows_written / total_rows

            if rows_written > 0:
                est_total_time = (
                    elapsed / (rows_written / total_rows)
                )

                est_remaining = (
                    est_total_time - elapsed
                ) / 60.0

            else:
                est_remaining = 0

            print(
                f"Chunk {chunk_idx + 1}/{n_chunks} - "
                f"{pct_done:.1f}% done, "
                f"{est_remaining:.1f} min remaining"
            )

        del chunk_links

    f.close()

    elapsed_total = time.time() - start_clock

    print(f'Finished with total time {elapsed_total}')