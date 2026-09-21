import numpy as np
import time
import pandas as pd
import networkx as nx



def generate_choiceset():
    links = pd.read_csv('input_link.csv')
    nodes = pd.read_csv('input_node.csv')

    zone_nodes = nodes[
        (nodes['zone_id'] > 0) & 
        (nodes['zone_id'] <= 405)
    ]

    zone_map = dict(zip(zone_nodes['zone_id'], zone_nodes['nodes_id']))

    agents = pd.read_csv('input_agent_peak_405.csv')

    od_pairs = (
        agents[['from_zone_id', 'to_zone_id']]
        .drop_duplicates()
        .to_numpy()
    )

    inter_od = od_pairs[
        od_pairs[:,0] != od_pairs[:,1]
    ]

    kb_link_ids = {
        285974,
        285975,
        285976,
        285977,
        299079,
        299080,
        299081
    }

    k_max = 10
    penalty_factor = 1.5
    max_overlap = 0.85
    max_candidates = 30

    def path_to_string(path):
        return ';'.join(str(x) for x in path)

    def compute_path_info(path, base_weights, link_idx_map):
        link_set = []
        travel_time = 0.0

        for i in range(len(path) - 1):
            key = f"{path[i]}_{path[i+1]}"

            if key not in link_idx_map:
                return False, 0.0, []

            idx = link_idx_map[key]

            travel_time += base_weights[idx]
            link_set.append(idx)

        return True, travel_time, link_set

    def overlap_fraction(link_set_1, link_set_2):
        common = len(set(link_set_1).intersection(link_set_2))

        max_len = max(len(link_set_1), len(link_set_2))

        if max_len == 0:
            return 0

        return common / max_len


    for network_type in [1,2]:
        if network_type == 1:
            links_use = links[~links['link_id'].isin(kb_link_ids)].copy()

            save_name = 'choice_set_noKB_k10_v2.npz'

        else:

            links_use = links.copy()

            save_name = 'choice_set_withKB_v2.npz'

    from_nodes = links_use['from_node_id'].to_numpy()
    to_nodes = links_use['to_node_id'].to_numpy()

    base_weights = links_use['free_flow_time'].to_numpy(dtype = float)

    link_lengths = links_use['length'].to_numpy(dtype = float)

    n_links_net = len(links_use)

    link_idx_map = {}

    for li in range(n_links_net):
        key = f"{from_nodes[li]}_{to_nodes[li]}"

        link_idx_map[key] = li

    finallist = [[None for _ in range(len(inter_od))] for _ in range(k_max)]

    routelocation = np.zeros((len(inter_od), 4), dtype = int)

    failed = 0

    stats_paths_found = np.zeros(len(inter_od))
    stats_candidates_generated = np.zeros(len(inter_od))
    stats_rejected_overlap = np.zeros(len(inter_od))

    start_time = time.time()

    for i in range(len(inter_od)):
        oz = inter_od[i, 0]
        dz = inter_od[i, 1]

        if oz not in zone_map or dz not in zone_map:
            failed += 1
            continue

        o_node = zone_map[oz]
        d_node = zone_map[dz]

        weights = base_weights.copy()

        accepted_paths = []
        accepted_linksets = []
        accepted_tts = []
        accepted_strs = []

        n_accepted = 0
        n_candidates = 0
        n_rejected = 0

        for candidate in range(max_candidates):

            if n_accepted >= k_max:
                break

            G = nx.DiGraph()

            for u, v, w in zip(from_nodes, to_nodes, weights):
                G.add_edge(u, v, weight = w)

            p = nx.shortest_path(
                G,
                source = o_node,
                target = d_node,
                weight = 'weight'
            )

            n_candidates += 1

            valid, real_tt, link_set = compute_path_info(p, base_weights, link_idx_map)

            if not valid:
                for j in range(len(p) - 1):
                    mask = (
                        (from_nodes == p[j]) & 
                        (to_nodes == p[j + 1])
                    )

                    weights[mask] *= penalty_factor

                continue

            accept = True

            for prev_links in accepted_linksets:
                overlap = overlap_fraction(link_set, prev_links)

                if overlap > max_overlap:
                    accept = False
                    n_rejected += 1
                    break

            if accept:
                accepted_paths.append(p)
                accepted_linksets.append(link_set)
                accepted_tts.append(real_tt)
                accepted_strs.append(path_to_string(p))

                n_accepted += 1

            for j in range(len(p) - 1 ):

                mask = (
                    (from_nodes == p[j]) & 
                    (to_nodes == p[j+1])
                )

                weights[mask] *= penalty_factor

        if n_accepted > 0:

            for k in range(n_accepted):
                finallist[k][i] = accepted_strs[k]

            routelocation[i, :] = [oz, dz, 1, i]

        else:
            failed += 1

        stats_paths_found[i] = n_accepted
        stats_candidates_generated[i] = n_candidates
        stats_rejected_overlap[i] = n_rejected

        if (i + 1) % 2000 == 0:
            elapsed = time.time - start_time

            remaining = (elapsed / (i + 1)) * (len(inter_od) - (i + 1))

            print(
                f"    {i + 1} / {len(inter_od)} done"
                f"({elapsed/60:.1f} min elapsed)"
                f"~{remaining/60:.1f} min remaining"
            )
        elapsed = time.time - start_time

        good = [
            idx for idx in range(len(finallist[0])) if finallist[0][idx] is not None
        ]

        finallist = [
            [row[idx] for idx in good] for row in finallist
        ]

        routelocation = routelocation[good]
        stats_paths_found = stats_paths_found[good]

        np.savez_compressed(
            save_name,
            finallist = np.array(finallist, dtype = object),
            routeloation = routelocation
        )

        print(f"\nSaved {save_name}")