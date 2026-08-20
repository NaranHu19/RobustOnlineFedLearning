import math


def k_schedule(
    nb_tot_update: int,
    aggreg_freq_scale: float,
    aggreg_mult_scale: float,
) -> list[int]:
    """
    Generate aggregation times.

    The aggregation times increase according to the specified growth factor until
    the total number of updates is reached.

    Parameters
    ----------
    nb_steps : int
        Total number of updates.
    aggreg_freq_scale : float
        Scaling of the aggregation frequency.
    aggreg_freq_scale : float
        Multiplicative scaling parameter.

    Returns
    -------
    list[int]
        Sequence of aggregation times.
    """
    next_time = 0
    step = 0
    schedule: list[int] = []

    while next_time < nb_tot_update:
        next_time = math.ceil(aggreg_mult_scale*step**aggreg_freq_scale)

        if next_time >= nb_tot_update:
            break

        schedule.append(next_time)
        step += 1

    schedule.append(nb_tot_update)

    return schedule
