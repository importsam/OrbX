__all__ = ["cluster", "synthetic_orbit", "density"]


def __getattr__(name):
    if name == "cluster":
        from orbx.clustering import cluster as _cluster

        globals()["cluster"] = _cluster
        return _cluster
    if name == "synthetic_orbit":
        from orbx.synthetic_orbits import synthetic_orbit as _synthetic_orbit

        globals()["synthetic_orbit"] = _synthetic_orbit
        return _synthetic_orbit
    if name == "density":

        from orbx.density.Density import density as _density

        globals()["density"] = _density
        return _density
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
