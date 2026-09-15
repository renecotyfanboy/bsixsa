"""SIXSA backend and its independently testable inference components."""

__all__ = ["SIXSABackend"]


def __getattr__(name):
    if name == "SIXSABackend":
        from ._sixsa import SIXSABackend

        return SIXSABackend
    raise AttributeError(name)
