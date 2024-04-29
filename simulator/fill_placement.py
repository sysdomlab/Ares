from policy.applications import APPLICATIONS


def get_placement_id(num_replicas):
    placement = ()
    while sum(placement) < num_replicas:
        placement = (*placement, min(num_replicas - sum(placement), 4))
    placement = tuple(filter(None, placement))
    placement = min(placement[i:] + placement[:i]
                    for i in range(len(placement)))
    placement_id = int("".join(map(str, placement)))
    return placement_id


if __name__ == '__main__':
    for app_name, app in APPLICATIONS.items():

        placement_max = app.placements[app.placements.local_bsz == app.max_local_bsz]
        print(f"app_name: {app_name}, local_bsz: {app.max_local_bsz}")
        print(placement_max)

        for i in range(1, 16):
            # 找到placement_max.placement中不存在的i
            if i not in placement_max.placement.values:
                # print(f"placement {get_placement_id(i)} not found")
                pass
