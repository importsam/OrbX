(This repo currently under construction)

# OrbX

This is a repository containing to tools for the paper:

> **OrbX: A Framework for Orbital Capacity Management**  
> Sam White, Samya Bagchi, Yasir Latif  
> 12th Annual Space Traffic Management Conference, Austin TX, February 2026  
> [link](https://utexas.app.box.com/v/26space-traffic-conference/file/2126683483061)

This repository provides two tools for generating synthetic reference orbits within an orbital cluster, built on top of the Kholshevnikov orbital distance metric and HDBSCAN-based clustering.

---

## Tools

### Fréchet Mean Orbit
Computes the orbit that minimises the mean squared Kholshevnikov distance to all members of a cluster — a generalization of the centroid to non-Euclidean metric spaces. Useful for siting a central reference asset (e.g. a refuelling or servicing satellite) that minimises access cost to all cluster members.

### Maximally Separated Orbit
Finds the orbit within the cluster's bounding region that maximises its minimum distance to any existing member — the "biggest gap" in the orbital neighbourhood. Useful for launch planning or manoeuvre targeting to reduce local congestion without leaving the operational regime.

---

## Demos

| Demo | Description |
|---|---|
| [Synthetic Orbit Cesium Model](<http://54.252.232.0:5000/>) | 3D Cesium visualisation of the Fréchet mean and maximally separated synthetic orbits within a cluster |
| [Unique Orbits (k-NN) Cesium Model](<https://orbx.spaceprotocol.org/>) | 3D Cesium visualisation of orbital neighborhoods and uniqueness across regimes |

---

