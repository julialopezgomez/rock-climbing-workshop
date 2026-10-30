"""GPU inference requires capacity confirmation by the public launcher."""


def require_gpu(confirmed):
    if not confirmed:
        raise ValueError(
            "Use scripts/perception/run.py so GPU capacity is checked before inference"
        )
