# Plane Wave Basis Sets for Twist Bilayer Graphene

## Graphene Coordinates

For TB model, see the notes in `/notes` folder. For continuum model, refer to [this PRX paper by Koshino](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031087)

## Code Structures

- [x] Tight Binding Model (CPU)

- [x] Tight Binding Model in Magnetic Field (Periodic Landau Gauge)

- [x] Continuum Model

## Dependent Module

See `requirements.txt`. You can refer to `.github/workflows/main.yml` I provide to set up the environment.

## Relationship between n_moire and angle

This is an output of wanniertools (full tb solution)
![nmoire](fig/nmoire_angle.png)

## Glist Construction

for tight binding (nsymm case, specific valley 1)

![](fig/glist_tb_v_1.png)

for magnetic tight binding (specific valley 1)

![](fig/glist_mtb_v_1.png)

## Example Output

### Tight Binding

For `n_moire = 30, valley = 1`:

![eg1](fig/band_n_30_v_1.png)

For `n_moire = 30, valley = -1`:

![eg2](fig/band_n_30_v_-1.png)

### Continuum

For `n_moire = 30, valley = 1`:

![eg3](fig/continuum.png)

For `n_moire = 30, valley = -1`:

![eg4](fig/continuum-1.png)
