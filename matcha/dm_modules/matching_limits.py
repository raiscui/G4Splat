DEFAULT_MATCHING_MAX_PAIRWISE_ELEMENTS = 100_000_000


def estimate_matching_pairwise_elements(n_charts: int, height: int, width: int) -> int:
    return int(n_charts) * int(n_charts) * int(height) * int(width)


def estimate_matching_tensor_gib(
    pairwise_elements: int,
    *,
    channels: int = 1,
    bytes_per_value: int = 4,
) -> float:
    return pairwise_elements * channels * bytes_per_value / (1024 ** 3)


def matching_loss_is_safe(
    n_charts: int,
    height: int,
    width: int,
    *,
    max_pairwise_elements: int | None = DEFAULT_MATCHING_MAX_PAIRWISE_ELEMENTS,
) -> tuple[bool, int]:
    pairwise_elements = estimate_matching_pairwise_elements(n_charts, height, width)
    if max_pairwise_elements is None:
        return True, pairwise_elements
    return pairwise_elements <= int(max_pairwise_elements), pairwise_elements
