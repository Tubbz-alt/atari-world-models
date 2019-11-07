def spread(n, number_of_bins):
    """ Split *n* into *number_of_bins* almost equally sized numbers.

    This always holds: sum(spread(n, number_of_bins)) == n
    """
    count, remaining = divmod(n, number_of_bins)
    result = [count] * number_of_bins
    for i in range(remaining):
        result[i] += 1
    return result
