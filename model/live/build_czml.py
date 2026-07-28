"""CZML builder for frozen mean-element orbit rings only."""

import numpy as np
import datetime as dt
import os
import json

from sgp4.api import Satrec
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import (
    TEME,
    ITRS,
    CartesianRepresentation,
)


# SGP4 uses the WGS-72 gravitational parameter.
MU_WGS72_KM3_S2 = 398600.8


def satrec_epoch_utc(satrec: Satrec) -> dt.datetime:
    year = 2000 + satrec.epochyr if satrec.epochyr < 100 else satrec.epochyr
    return dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=satrec.epochdays - 1
    )


def _fmt_czml_time(when: dt.datetime) -> str:
    return when.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def teme_vector_to_frozen_itrs_metres(
    position_teme_km,
    reference_utc: dt.datetime,
):
    """
    Static visual orbit geometry.

    Interpret a TEME-like cartesian vector in axes frozen at reference_utc,
    then convert once into the corresponding fixed Earth frame.
    """
    reference_time = Time(reference_utc)

    frozen_teme = TEME(
        CartesianRepresentation(
            position_teme_km[0] * u.km,
            position_teme_km[1] * u.km,
            position_teme_km[2] * u.km,
        ),
        obstime=reference_time,
    )

    frozen_itrs = frozen_teme.transform_to(ITRS(obstime=reference_time))
    xyz_m = frozen_itrs.cartesian.xyz.to_value(u.m)
    return float(xyz_m[0]), float(xyz_m[1]), float(xyz_m[2])


def orbital_period_seconds(satrec: Satrec) -> float:
    """Nominal Keplerian period from Kozai mean motion (seconds, not truncated)."""
    return float((2.0 * np.pi / satrec.no_kozai) * 60.0)


def sample_frozen_orbit_ring(
    tle_line1,
    tle_line2,
    t_ref: dt.datetime,
    norad_id: str = "",
):
    """
    Exact closed mean-Keplerian orbit ring.

    Static shape-comparison visual from the TLE's mean elements (not an
    SGP4 time-propagated track). All vertices use TEME-like axes frozen at
    ``t_ref``, then transform once to ITRS at that same epoch.

    """
    satrec = Satrec.twoline2rv(tle_line1, tle_line2)

    # no_kozai: radians per minute in the SGP4 library.
    n_rad_s = float(satrec.no_kozai) / 60.0
    if n_rad_s <= 0.0:
        raise ValueError(f"{norad_id or 'ring'}: invalid mean motion")

    # TLE mean semi-major axis, km.
    a_km = (MU_WGS72_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
    e = float(satrec.ecco)

    if not (0.0 <= e < 1.0):
        raise ValueError(f"{norad_id or 'ring'}: invalid eccentricity {e}")

    inc = float(satrec.inclo)
    raan = float(satrec.nodeo)
    argp = float(satrec.argpo)

    # Visual density only
    vertex_count = 120
    eccentric_anomaly = np.linspace(
        0.0,
        2.0 * np.pi,
        vertex_count,
        endpoint=False,
    )

    # Ellipse in perifocal coordinates, km.
    b_km = a_km * np.sqrt(1.0 - e * e)
    x_pf = a_km * (np.cos(eccentric_anomaly) - e)
    y_pf = b_km * np.sin(eccentric_anomaly)

    # Perifocal -> TEME-like inertial axes:
    # R3(RAAN) @ R1(inclination) @ R3(argument of perigee)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    rotation = np.array(
        [
            [
                cO * cw - sO * sw * ci,
                -cO * sw - sO * cw * ci,
                sO * si,
            ],
            [
                sO * cw + cO * sw * ci,
                -sO * sw + cO * cw * ci,
                -cO * si,
            ],
            [
                sw * si,
                cw * si,
                ci,
            ],
        ]
    )

    perifocal_xyz_km = np.vstack(
        (
            x_pf,
            y_pf,
            np.zeros_like(x_pf),
        )
    )

    inertial_xyz_km = rotation @ perifocal_xyz_km

    ring = []
    for xyz_km in inertial_xyz_km.T:
        x_m, y_m, z_m = teme_vector_to_frozen_itrs_metres(
            xyz_km,
            reference_utc=t_ref,
        )
        ring.extend([x_m, y_m, z_m])

    # Exact duplicate of vertex zero: mathematically and numerically closed.
    ring.extend(ring[:3])

    print(
        f"{norad_id or 'ring'}: "
        f"a={a_km:.3f} km, e={e:.8f}, "
        f"vertices={vertex_count}, mean-element ring"
    )

    return ring


def _validate_cartesian_ring(samples):
    coords = []
    for coord in samples:
        value = float(coord)
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Invalid ring coordinate: {value}")
        coords.append(value)
    return coords


def _synthetic_type_of(row) -> str:
    st = row.get("synthetic_type", None)
    if st is None or (isinstance(st, float) and np.isnan(st)):
        return "None"
    return str(st)


def global_display_epoch(df):
    epochs = []
    for _, row in df.iterrows():
        sat = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
        epochs.append(satrec_epoch_utc(sat))

    mean_ts = float(np.mean([e.timestamp() for e in epochs]))
    return dt.datetime.fromtimestamp(mean_ts, tz=dt.timezone.utc)


def cluster_display_epoch_and_period(group):
    epochs = []
    periods = []
    frechet_ep = frechet_period = None
    maxsep_ep = maxsep_period = None

    for _, row in group.iterrows():
        sat = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
        ep = satrec_epoch_utc(sat)
        per = orbital_period_seconds(sat)
        epochs.append(ep)
        periods.append(per)
        st = _synthetic_type_of(row)
        if st == "frechet":
            frechet_ep, frechet_period = ep, per
        elif st == "max_separation":
            maxsep_ep, maxsep_period = ep, per

    if frechet_ep is not None:
        return frechet_ep, frechet_period
    if maxsep_ep is not None:
        return maxsep_ep, maxsep_period

    mean_ts = float(np.mean([e.timestamp() for e in epochs]))
    t_display = dt.datetime.fromtimestamp(mean_ts, tz=dt.timezone.utc)
    return t_display, float(max(periods))


def build_czml(df):
    import pandas as pd

    print("Building CZML: frozen mean-element orbit rings only")

    czml = [{"id": "document", "version": "1.0"}]
    property_keys = [k for k in df.columns if k.startswith("prop_")]
    t_display_global = global_display_epoch(df)
    label_col = "label" if "label" in df.columns else "cluster_label"

    for _, row in df.iterrows():
        norad_id = str(row["NORAD_CAT_ID"])
        try:
            raw_lab = row.get(label_col, -1)
            t_display = t_display_global

            ring = sample_frozen_orbit_ring(
                row["TLE_LINE1"],
                row["TLE_LINE2"],
                t_display,
                norad_id=str(row.get("NORAD_CAT_ID", "")),
            )
            ring_coords = _validate_cartesian_ring(ring)

            epoch_str = _fmt_czml_time(t_display)

            neighbours = row.get("neighbours", [])
            neighbours_dict = {}
            for i, neighbour in enumerate(neighbours):
                if isinstance(neighbour, (str, int, float, bool, type(None))):
                    neighbours_dict[str(i + 1)] = neighbour
                else:
                    neighbours_dict[str(i + 1)] = str(neighbour)

            cluster_label = str(raw_lab)
            synthetic_type = _synthetic_type_of(row)

            cluster_density = row.get("cluster_density", None)
            if cluster_density is None or (
                isinstance(cluster_density, float) and np.isnan(cluster_density)
            ):
                cluster_density = "None"
            else:
                cluster_density = float(cluster_density)

            sat_native = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
            tle_epoch_str = _fmt_czml_time(satrec_epoch_utc(sat_native))

            additional_properties = {
                "uniqueness_range": row.get("uniqueness_range", "none"),
                "neighbours": neighbours_dict,
                "cluster_label": cluster_label,
                "synthetic_type": synthetic_type,
                "cluster_density": cluster_density,
                "tle_epoch": tle_epoch_str,
                "display_epoch": epoch_str,
            }

            cleaned_properties = {}
            for key in property_keys:
                prop_value = row.get(key)
                short_key = key[5:]
                if prop_value is None:
                    cleaned_properties[short_key] = None
                elif isinstance(prop_value, (np.floating, float)):
                    prop_value = float(prop_value)
                    if np.isnan(prop_value) or np.isinf(prop_value):
                        cleaned_properties[short_key] = None
                    else:
                        cleaned_properties[short_key] = prop_value
                elif isinstance(prop_value, (np.integer, int)):
                    cleaned_properties[short_key] = int(prop_value)
                elif isinstance(prop_value, (str, bool)):
                    cleaned_properties[short_key] = prop_value
                else:
                    cleaned_properties[short_key] = str(prop_value)

            if synthetic_type == "frechet":
                ring_rgba = [255, 0, 0, 255]
            elif synthetic_type == "max_separation":
                ring_rgba = [0, 100, 255, 255]
            else:
                ring_rgba = [32, 201, 151, 255]

            object_name = str(row["OBJECT_NAME"])
            is_synthetic = synthetic_type in ("frechet", "max_separation")

            if not is_synthetic:
                # Keep a lightweight lookup entity so the frontend can still
                # search by bare NORAD id and map it to the orbit ring.
                czml.append(
                    {
                        "id": norad_id,
                        "name": object_name,
                        "properties": {**cleaned_properties, **additional_properties},
                    }
                )

            # Orbit ring for shape comparison (no availability — always drawable).
            # Synthetics use the bare SYN_* id (no -orbit-ring suffix).
            ring_id = norad_id if is_synthetic else f"{norad_id}-orbit-ring"
            czml.append(
                {
                    "id": ring_id,
                    "name": object_name,
                    "polyline": {
                        "positions": {"cartesian": ring_coords},
                        "width": 2,
                        "material": {
                            "solidColor": {"color": {"rgba": ring_rgba}}
                        },
                        "arcType": "NONE",
                        "clampToGround": False,
                        "show": False,
                    },
                    "properties": {
                        "cluster_label": cluster_label,
                        "synthetic_type": synthetic_type,
                        "parent_norad": norad_id,
                        # Rings are what Cesium picks on hover; keep uniqueness
                        # metadata on them as well as on the bare NORAD entity.
                        **cleaned_properties,
                        **additional_properties,
                    },
                }
            )

        except Exception as e:
            print(f"Error processing satellite {norad_id}: {e}")
            continue

    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "output.czml")

    try:
        with open(output_file, "w") as file:
            try:
                import simplejson as json_module
            except ImportError:
                import json as json_module

            json_module.dump(czml, file, indent=2, separators=(",", ": "))

        with open(output_file, "r") as file:
            _ = json_module.load(file)
            print("CZML file validated successfully")

    except Exception as e:
        print(f"Error writing or validating CZML file: {e}")
