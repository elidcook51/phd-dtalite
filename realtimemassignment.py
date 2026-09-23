import numpy as np
import pandas as pd
from scipy.io import savemat
from traveltimecal import traveltimecal, traveltimecal_fast

def realtimeassignment(itr, choiceset, routelocation, phlength, realtime_user, nchoice, rposition, bigloop, dsize, num_tdlink_rows):
    onlylike0 = []
    onlylike1 = []

    ttt = 0


    S = pd.read_csv('DTALite_Files/input_agent_initial.csv')

    agent = np.zeros((len(S), 2))
    agent[:, 0] = S.iloc[:,0].to_numpy()
    agent[:, 1] = S.iloc[:,6].to_numpy()

    agentOD = np.zeros((len(S), 2))
    agentOD[:,0] = S.iloc[:, 4].to_numpy()
    agentOD[:,1] = S.iloc[:, 5].to_numpy()

    if itr != 1:

        T = pd.read_csv('output_agent.csv')

        agent_tt = T.iloc[:, 12].to_numpy()

        agentpath = T.iloc[:, 29].tolist()

    if itr == 1:
        TDlink = np.column_stack(
            [pd.read_excel(
                'DTALite_Files/Each iteration.csv',
                sheet_name = 0,
                usecol = 'A',
                skiprows = 1,
                nrows = num_tdlink_rows,
            ).to_numpy().flatten(),
            pd.read_excel(
                'DTALite_Files/Each iteration.csv',
                sheet_name = 0,
                usecol = 'B',
                skiprows = 1,
                nrows = num_tdlink_rows,
            ).to_numpy().flatten(),
            pd.read_excel(
                'DTALite_Files/Each iteration.csv',
                sheet_name = 0,
                usecol = 'E',
                skiprows = 1,
                nrows = num_tdlink_rows,
            ).to_numpy().flatten(),
            pd.read_excel(
                'DTALite_Files/Each iteration.csv',
                sheet_name = 0,
                usecol = 'F',
                skiprows = 1,
                nrows = num_tdlink_rows,
            ).to_numpy().flatten(),]
        )
    else:
        TDlink = np.column_stack([
            pd.read_excel(
                'output_linkTMOE.csv',
                sheet_name = 0,
                usecols = 'A',
                skiprows = 1,
                nrows = num_tdlink_rows,
            ).to_numpy().flatten(),
            pd.read_excel(
                'output_linkTMOE.csv',
                sheet_name = 0,
                usecols = 'B',
                skiprows = 1,
                nrows = num_tdlink_rows,
            ).to_numpy().flatten(),
            pd.read_excel(
                'output_linkTMOE.csv',
                sheet_name = 0,
                usecols = 'E',
                skiprows = 1,
                nrows = num_tdlink_rows,
            ).to_numpy().flatten(),
            pd.read_excel(
                'output_linkTMOE.csv',
                sheet_name = 0,
                usecols = 'F',
                skiprows = 1,
                nrows = num_tdlink_rows,
            ).to_numpy().flatten(),
        ])

    b = TDlink[:, 0]

    tdlink_map = {}

    for r in range(TDlink.shape[0]):
        key = f"{int(TDlink[r,2])}_{int(TDlink[r,0])}_{int(TDlink[r,1])}"
        tdlink_map[key] = TDlink[r,3]

    gas = 3.0
    links = pd.read_csv('DTALite_Files/input_link.csv')

    length = links[['from_node_id', 'to_node_id', 'length', 'free_speed']].to_numpy()

    len_arr = np.zeros((len(choiceset), len(choiceset[0])))
    nc = np.zeros((len(choiceset), len(choiceset[0])))

    for h in range(len(choiceset[0])):
        for i in range(len(choiceset)):

            if not choiceset[i]:
                continue

            sque = choiceset[i][h].split(';')

            index = []
            for j in range(len(sque) - 1):
                index.append(float(sque[j]))

                index = np.array(index)

                for j in range(len(index) - 1):
                    pl = np.where(
                        (length[:,0] == index[j]) & 
                        (length[:,1] == index[j + 1])
                    )[0]

                    if len(pl) > 0:
                        len_arr[i, h] += length[pl[0], 2]

            nc[i, h] = len(index) - 1

    gas = 3

    tt = [[None for _ in range(len(choiceset[0]))] for _ in range(len(choiceset))]

    pltt = [[None for _ in range(len(choiceset[0]))] for _ in range(len(choiceset))]

    fuelcost = [[None for _ in range(len(choiceset[0]))] for _ in range(len(choiceset))]

    for h in range(len(choiceset[0])):
        for i in range(len(choiceset)):

            path = choiceset[i][h]

            if not path:
                continue

            m = 1
            findagent = []
            agentn = []

            if itr == 1:
                m = 2

            else:
                findagent = [
                    idx for idx, p in enumerate(agentpath) if p == path
                ]

                for idx in findagent:

                    agentn.append(
                        [agent[idx,0],
                         agent[idx,1],
                         agent_tt[idx]]
                    )

                    m += 1

                agentn = np.array(agentn) if agentn else np.empty((0,3))

            for k in range(
                itr * phlength, itr * phlength + 1,
            ):

                if m == 1:

                    tt_val, fuel_val = traveltimecal_fast(
                        k + 899,
                        TDlink,
                        choiceset[i][h],
                        length,
                        gas,
                        itr,
                        0
                    )

                    if tt[i][h] is None:
                        tt[i][h] = {}
                    if pltt[i][h] is None:
                        pltt[i][h] = {}
                    if fuelcost[i][h] is None:
                        fuelcost[i][h] = {}

                    tt[i][h][k] = tt_val
                    pltt[i][h][k] = tt_val
                    fuelcost[i][h][k] = fuel_val

                else:

                    if itr == 1:
                        ttloc = np.array([])

                    else:
                        ttloc = np.where(
                            np.floor(agentn[:, 1] == (k + 899))
                        )[0]

                    if len(ttloc) == 0:

                        tt_val, fuel_val = traveltimecal_fast(
                            k + 899,
                            TDlink,
                            choiceset[i][h],
                            length,
                            gas,
                            itr,
                            0
                        )

                        if tt[i][h] is None:
                            tt[i][h] = {}
                        if pltt[i][h] is None:
                            pltt[i][h] = {}
                        if fuelcost[i][h] is None:
                            fuelcost[i][h] = {}

                        tt[i][h][k] = tt_val
                        pltt[i][h][k] = tt_val
                        fuelcost[i][h][k] = fuel_val

                    else:
                        a = [agentn[idx, 2] for idx in ttloc]

                        if tt[i][h] is None:
                            tt[i][h] = {}
                        if pltt[i][h] is None:
                            pltt[i][h] = {}
                        if fuelcost[i][h] is None:
                            fuelcost[i][h] = {}

                        tt[i][h][k] = np.mean(a)
                        pltt[i][h][k] = np.max(a)

                        _, fuel_val = traveltimecal_fast(
                            k + 899,
                            TDlink,
                            choiceset[i][h],
                            length,
                            gas,
                            itr,
                            0
                        )

                        fuelcost[i][h][k] = fuel_val

    mask = agent[:, 1] <= (360 + phlength * itr)
    p_agent = agent[mask]

    mask2 = (360 + phlength * (itr - 1)) < p_agent[:, 1]
    p_agent = p_agent[mask2]

    rtchoice = np.zeros((p_agent.shape[0], 1))
    updaterposition = np.zeros((p_agent.shape[0], 1))

    for i in range(p_agent.shape[0]):
        agent_id = int(p_agent[i, 0])

        if agent_id not in realtime_user:

            rtchoice[i, 0] = nchoice[agent_id + 1, 0]
            updaterposition[i, 0] = rposition[agent_id + 1, 0]

        else:

            ff = np.where(
                (routelocation[:, 0] == agentOD[agent_id + 1, 0]) & 
                (routelocation[:, 1] == agentOD[agent_id + 1, 1])
            )[0]

            f = int(routelocation[ff[0], 3])

            po = itr * phlength

            b1 = [
                len_arr[0, f],
                tt[0][f][po],
                pltt[0][f][po],
                fuelcost[0][f][po],
                nc[0, f],
            ]

            croute = 1

            for j in range(1, len(choiceset)):

                if not choiceset[j]:
                    continue

                b0 = [
                    len_arr[j, f],
                    tt[j][f][po],
                    pltt[j][f][po],
                    fuelcost[j][f][po],
                    nc[j,f]
                ]

                if b0[1] < b1:
                    choice = 0
                else:
                    choice = 1

                if choice == 0:
                    b1 = b0
                    croute = j + 1

            rtchoice[i, 0] = croute
            updaterposition[i, 0] = f

    if itr == np.floor(60 / phlength):

        fname = f"pathinfo_realass{bigloop}.mat"

        savemat(
            fname,
            {
                'len': len_arr,
                'tt': tt,
                'pltt': pltt,
                'fuelcost': fuelcost,
                'nc': nc
            }
        )

    return rtchoice, updaterposition, tt, b, p_agent


def comrealtimeassignment(
    itr,
    choiceset,
    routelocation,
    phlength,
    realtime_user,
    nchoice,
    rposition,
    weights,
    meanstd28,
    meanstd2,
    user,
    bigloop,
    dsize,
    num_tdlink_rows,
):

    onlylike0 = []
    onlylike1 = []

    # --------------------------------------------------
    # Read agent information
    # --------------------------------------------------
    S = pd.read_csv("input_agent_initial.csv")

    agent = np.zeros((len(S), 2))
    agent[:, 0] = S.iloc[:, 0]
    agent[:, 1] = S.iloc[:, 6]

    agentOD = np.zeros((len(S), 2))
    agentOD[:, 0] = S.iloc[:, 4]
    agentOD[:, 1] = S.iloc[:, 5]

    if itr != 1:
        T = pd.read_csv("output_agent.csv")

        agent_tt = T.iloc[:, 12].to_numpy()
        agentpath = T.iloc[:, 29].astype(str).tolist()

    # --------------------------------------------------
    # Read TD link information
    # --------------------------------------------------
    if itr == 1:
        td_df = pd.read_excel(
            "Each iteration.csv",
            usecols=[0, 1, 4, 5]
        )
    else:
        td_df = pd.read_csv(
            "output_linkTDMOE.csv",
            usecols=[0, 1, 4, 5]
        )

    TDlink = td_df.to_numpy()

    b = TDlink[:, 0]

    # --------------------------------------------------
    # Build TD link hash map
    # --------------------------------------------------
    print("  Building TDlink hash map...")

    tdlink_map = {}

    for row in TDlink:
        key = f"{int(row[2])}_{int(row[0])}_{int(row[1])}"
        tdlink_map[key] = row[3]

    print(f"  ✓ Hash map with {len(tdlink_map)} entries")

    # --------------------------------------------------
    # Route attributes
    # --------------------------------------------------
    gas = 3.0
    links = pd.read_csv('input_link.csv')
    
    length_data = links[['from_node_id', 'to_node_id', 'length', 'free_speed']].to_numpy()

    n_routes = len(choiceset)
    n_ods = len(choiceset[0])

    len_mat = np.zeros((n_routes, n_ods))
    nc = np.zeros((n_routes, n_ods))

    for h in range(n_ods):

        for i in range(n_routes):

            path = choiceset[i][h]

            if not path:
                continue

            nodes = [
                int(x)
                for x in path.split(";")
                if x.strip() != ""
            ]

            for j in range(len(nodes) - 1):

                mask = (
                    (length_data[:, 0] == nodes[j])
                    &
                    (length_data[:, 1] == nodes[j + 1])
                )

                idx = np.where(mask)[0]

                if len(idx) > 0:
                    len_mat[i, h] += length_data[idx[0], 2]

            nc[i, h] = len(nodes) - 1

    gas = 3

    # --------------------------------------------------
    # Travel time calculation
    # --------------------------------------------------
    tt = {}
    pltt = {}
    fuelcost = {}

    for h in range(n_ods):

        for i in range(n_routes):

            path = choiceset[i][h]

            if not path:
                continue

            m = 1

            if itr == 1:

                m = 2

            else:

                findagent = [
                    idx
                    for idx, p in enumerate(agentpath)
                    if p == path
                ]

                agentn = []

                for fa in findagent:

                    agentn.append([
                        agent[fa, 0],
                        agent[fa, 1],
                        agent_tt[fa]
                    ])

                agentn = np.array(agentn)

                m = len(agentn) + 1

            for k in [itr * phlength]:

                if m == 1:

                    tt_val, fc_val = traveltimecal(
                        k + 899,
                        TDlink,
                        path,
                        length_data,
                        gas,
                        itr,
                        0,
                    )

                    tt[(i, h, k)] = tt_val
                    fuelcost[(i, h, k)] = fc_val
                    pltt[(i, h, k)] = tt_val

                else:

                    if itr == 1:
                        ttloc = []
                    else:

                        ttloc = np.where(
                            np.floor(agentn[:, 1]) == (k + 899)
                        )[0]

                    if len(ttloc) == 0:

                        tt_val, fc_val = traveltimecal(
                            k + 899,
                            TDlink,
                            path,
                            length_data,
                            gas,
                            itr,
                            0,
                        )

                        tt[(i, h, k)] = tt_val
                        fuelcost[(i, h, k)] = fc_val
                        pltt[(i, h, k)] = tt_val

                    else:

                        a = agentn[ttloc, 2]

                        tt[(i, h, k)] = np.mean(a)
                        pltt[(i, h, k)] = np.max(a)

                        _, fc_val = traveltimecal(
                            k + 899,
                            TDlink,
                            path,
                            length_data,
                            gas,
                            itr,
                            0,
                        )

                        fuelcost[(i, h, k)] = fc_val

    # --------------------------------------------------
    # Agents departing in current assignment period
    # --------------------------------------------------
    p_agent = agent[
        (agent[:, 1] <= (360 + phlength * itr))
        &
        (agent[:, 1] > (360 + phlength * (itr - 1)))
    ]

    rtchoice = np.zeros(len(p_agent), dtype=int)
    updaterposition = np.zeros(len(p_agent), dtype=int)
    pre_choice = np.zeros(len(p_agent), dtype=int)

    # --------------------------------------------------
    # Route choice
    # --------------------------------------------------
    for i in range(len(p_agent)):

        agent_id = int(p_agent[i, 0])

        if agent_id not in realtime_user:

            rtchoice[i] = nchoice[agent_id + 1]
            updaterposition[i] = rposition[agent_id + 1]

            continue

        ff = np.where(
            (routelocation[:, 0] ==
             agentOD[agent_id + 1, 0])
            &
            (routelocation[:, 1] ==
             agentOD[agent_id + 1, 1])
        )[0]

        f = int(routelocation[ff[0], 3])

        po = itr * phlength

        b1 = np.array([
            len_mat[0, f],
            tt[(0, f, po)],
            pltt[(0, f, po)],
            fuelcost[(0, f, po)],
            nc[0, f]
        ])

        pre_b1 = b1.copy()

        croute = 1
        pre_croute = 1

        for j in range(1, n_routes):

            if not choiceset[j]:
                continue

            b0 = np.array([
                len_mat[j, f],
                tt[(j, f, po)],
                pltt[(j, f, po)],
                fuelcost[(j, f, po)],
                nc[j, f]
            ])

            pre_b0 = b0.copy()

            bdiff = b1 - b0

            if i in onlylike1:

                choice = 0

            elif i in onlylike0:

                choice = 1

            else:

                if user[i, 1] < 16650:

                    bscale = bdiff.copy()

                    ojvalue = np.sum(
                        bscale * weights[int(user[i, 1]), :5]
                    )

                else:

                    bscale = (
                        bdiff
                        - meanstd2[0, :]
                    )

                    bscale = (
                        bscale
                        / meanstd2[1, :]
                    )

                    ojvalue = np.sum(
                        bscale * weights[int(user[i, 1]), :5]
                    )

                if abs(ojvalue - 1) > abs(ojvalue + 1):
                    choice = 0
                else:
                    choice = 1

            if choice != 1:
                b1 = b0
                croute = j + 1

            if pre_b1[1] >= pre_b0:
                pre_croute = j + 1
                pre_b1 = pre_b0

        rtchoice[i] = croute
        updaterposition[i] = f
        pre_choice[i] = pre_croute

    # --------------------------------------------------
    # Save final iteration route information
    # --------------------------------------------------
    if itr == int(np.floor(60 / phlength)):

        savemat(
            f"pathinfo_comrealass{bigloop}.mat",
            {
                "len": len_mat,
                "tt": tt,
                "pltt": pltt,
                "fuelcost": fuelcost,
                "nc": nc,
            },
        )

    return (
        rtchoice,
        updaterposition,
        tt,
        b,
        p_agent,
        pre_choice,
    )