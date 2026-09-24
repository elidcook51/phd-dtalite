import os
import shutil
import subprocess
from assignment import assignment, comassignment
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def dividerand(driver_ids, p, p_realtime, p_fix):

    rng = np.random.default_rng(seed = 1)

    driver_ids = np.array(driver_ids)
    rng.shuffle(driver_ids)

    n = len(driver_ids)

    n_member = int(round(p * n))
    n_realtime = int(round(p_realtime * n))
    n_fix = int(round(p_fix * n))

    member = driver_ids[:n_member]
    realtime_user = driver_ids[n_member:n_member + n_realtime]
    fix_user = driver_ids[n_member + n_realtime:n_member + n_realtime + n_fix]

    return member, realtime_user, fix_user

def msa(bigloop, p, p_fix, p_realtime):

    temp_agent_data = pd.read_csv('DTALite_Files/input_agent_initial.csv')

    inputagent = temp_agent_data.values.tolist()

    dsize = len(inputagent)

    temp_tdlink = pd.read_csv('DTALite_Files/Each iteration.csv')
    num_tdlink_rows = len(temp_tdlink) - 1

    driver_ids = np.arange(dsize)
    member, realtime_user, fix_user = dividerand(driver_ids, p, p_realtime, p_fix)

    mat = loadmat('DTALite_Files/choice set no overlap try.mat', simplify_cells = True)

    choiceset = mat['finallist'].T

    routelocation = []

    for row_idx in range(choiceset.shape[0]):
        for col_idx in range(choiceset.shape[1]):

            route_string = choiceset[row_idx, col_idx]

            if not isinstance(route_string, str):
                continue

            parts = route_string.strip(';').split(';')

            o = float(parts[0])
            d = float(parts[-1])

            routelocation.append([o,d,1,col_idx,row_idx])

    routelocation = np.array(routelocation)

    user = pd.read_excel(
        'DTALite_Files/SiouxFalls_net.xlsx',
        sheet_name=4,
        usecols = 'A:B',
        nrows = dsize
    ).values

    weights = pd.read_excel(
        'DTALite_Files/SiouxFalls_net.xlsx',
        sheet_name=5,
        usecols = 'A:F',
        skiprows = 5
    ).values

    realweights = pd.read_excel(
        'DTALite_Files/SiouxFalls_net.xlsx',
        sheet_name=5,
        usecols = 'J',
        skiprows = 5
    ).values

    meanstd28 = pd.read_excel(
        'DTALite_Files/SiouxFalls_net.xlsx',
        sheet_name =5,
        usecols = 'A:E',
        skiprows = 1,
        nrows = 2
    ).values

    meanstd2 = pd.read_excel(
        'DTALite_Files/SiouxFalls_net.xlsx',
        sheet_name = 5,
        usecols = 'H:L',
        skiprows = 1,
        nrows = 1
    ).values

    a0 = np.zeros((dsize, 3))

    for i in range(dsize):
        a0[i, 0] = inputagent[i][6]
        a0[i, 1] = inputagent[i][4]
        a0[i, 2] = inputagent[i][5]

    a = np.floor(a0)

    intervalID = np.zeros(len(choiceset))

    inform = []
    rposition = []

    cc = np.zeros(25)

    linkV = []
    rprime = {}

    prev_choice = None

    for itr in range(1, 3):
        print(f"Iteration {itr}")
        print(f"Starting running assignment")

        (
            choice,
            rposition,
            tt,
            ttt,
            inform
        ) = assignment(
            itr,
            choiceset,
            user,
            routelocation,
            weights,
            meanstd28,
            meanstd2,
            member,
            fix_user,
            realtime_user,
            a0,
            inputagent,
            bigloop,
            dsize,
            num_tdlink_rows
        )

        bprime = np.column_stack([
            choice,
            a[:,0],
            a[:,1],
            a[:,2]
        ])

        for h, route_string in enumerate(choiceset):
            route_string = str(route_string)

            parts = route_string.split(';')

            o = float(parts[0])
            d = float(parts[-1])

            for route_num in np.unique(choice):

                for interval in range(60):

                    idx = np.where(
                        (bprime[:, 1] == interval + 359) & 
                        (bprime[:, 2] == o) & 
                        (bprime[:, 3] == d) & 
                        (bprime[:, 0] == route_num)
                    )[0]

                    rprime[
                        (
                            int(route_num),
                            h,
                            interval,
                            itr
                        )
                    ] = len(idx)

        for j in range(dsize):
            route_idx = int(rposition[j])

            inputagent[j][11] = choiceset[
                route_idx
            ]

        columns = ['agent_id', 'tour_id', 'from_zone_id', 'to_zone_id', 'from_origin_node_id', 'to_destination_node_id', 'departure_time_in_min','demand_type', 'PCE', 'infomration_type', 'vehicle_age', 'path_node_sequencing', 'vehicle_type', 'pricing_type', 'value_of_time']

        pd.DataFrame(
            inputagent,
            columns =columns
        ).to_csv(
            'input_agent.csv', index = False
        )

        subprocess.run(
            [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True
        )

        shutil.copyfile(
            'DTALite_Files/output_LinkTDMOE.csv',
            'DTALite_Files/iteration_LinkTDMOE.csv'
        )

        if itr > 1:
            cc[itr - 1] = np.sum(choice != prev_choice)

        prev_choice = np.array(choice)

        dd = pd.read_csv('DTALite_Files/output_LinkTDMOE.csv')

        linkV.append(dd.iloc[:, 7].to_numpy())

    shutil.copyfile(
        'DTALite_Files/output_agent.csv',
        f"DTALite_Files/{bigloop}UOinfor_agent.csv"
    )

    shutil.copyfile(
        'DTALite_Files/output_LinkMOE.csv',
        f"DTALite_Files/{bigloop}UOinfor_LinkMOE.csv"
    )

    shutil.copyfile(
        'DTALite_Files/output_LinkTDMOE.csv',
        f"DTALite_Files/{bigloop}UOinfor_LinkTDMOE.csv"
    )

    shutil.copyfile(
        'DTALite_Files/output_agent.csv',
        'DTALite_Files/UOinfor_agent.csv'
    )

    shutil.copyfile(
        'DTALite_Files/output_LinkMOE.csv',
        'DTALite_Files/UOinfor_LinkMOE.csv'
    )

    shutil.copyfile(
        'DTALite_Files/output_LinkTDMOE.csv',
        'DTALite_Files/UOinfor_LinkTDMOE.csv'
    )

    (
        pre_choice,
        pre_rposition,
        pre_tt,
        pre_ttt,
        pre_inform
    ) = assignment(
        2,
        choiceset,
        user,
        routelocation,
        weights,
        meanstd28,
        meanstd2,
        member,
        fix_user,
        realtime_user,
        a0,
        inputagent,
        bigloop,
        dsize,
        num_tdlink_rows
    )

    for j in range(dsize):

        inputagent[j][11] = choiceset[
            int(pre_rposition[j])
        ]

    pd.DataFrame(inputagent, columns = columns).to_csv('input_agent.csv')

    subprocess.run(
        [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True
    )

    shutil.copyfile(
        'DTALite_Files/output_agent.csv',
        f"DTALite_Files/{bigloop}predict_agent.csv"
    )

    shutil.copyfile(
        'DTALite_Files/output_LinkMOE.csv',
        f"DTALite_Files/{bigloop}predict_LinkMOE.csv"
    )

    shutil.copyfile(
        'DTALite_Files/output_LinkTDMOE.csv',
        f"DTALite_Files/{bigloop}predict_LinkTDMOE.csv"
    )

    (
        com_choice,
        com_rposition,
        com_tt,
        com_ttt,
        com_inform,
        pre_choice
    ) = comassignment(
        100,
        choiceset,
        user,
        routelocation,
        realweights,
        meanstd28,
        meanstd2,
        member,
        fix_user,
        realtime_user,
        a0,
        inputagent,
        bigloop,
        dsize,
        num_tdlink_rows
    )

    for j in range(dsize):

        inputagent[j][11] = choiceset[
            int(com_rposition[j])
        ]

    pd.DataFrame(inputagent, columns=columns).to_csv('input_agent.csv', index = False)

    subprocess.run(
        [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True
    )

    shutil.copyfile(
        'DTALite_Files/output_agent.csv',
        f"DTALite_Files/{bigloop}actual_agent.csv"
    )

    shutil.copyfile(
        'DTALite_Files/output_LinkMOE.csv',
        f"DTALite_Files/{bigloop}actual_LinkMOE.csv"
    )

    shutil.copyfile(
        'DTALite_Files/output_LinkTDMOE.csv',
        f"DTALite_Files/{bigloop}actual_LinkTDMOE.csv"
    )

    (
        com_choiceOK,
        com_rpositionOK,
        com_ttOK,
        com_tttOK,
        pre_choiceOK
    ) = comassignment(
        2,
        choiceset,
        user,
        routelocation,
        realweights,
        meanstd28,
        meanstd2,
        member,
        fix_user,
        realtime_user,
        a0,
        inputagent,
        bigloop + 100,
        dsize,
        num_tdlink_rows
    )

    savemat(
        f"DTALite_Files/myfile{bigloop}.mat",
        {
            'cc': cc,
            'rprime': rprime,
            'linkV': np.array(linkV, dypte = object),
            'choice': pre_choice
        }
    )

