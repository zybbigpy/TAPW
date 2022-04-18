# Tight Binding Planewave Method for Twisted Bilayer Graphene

## Reference and Citation

1. For continuum model, refer to [this PRX paper by Koshino](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031087).
2. For TB Planewave method, please cite my MPhil Thesis.

## Code Structures

- [x] Tight Binding Model (CPU)
- [x] Tight Binding for Relaxed Strucutre 

- [x] Continuum Model

## Dependent Module

See `requirements.txt`. You can refer to `.github/workflows/main.yml` I provide to set up the environment.

## Relationship between n_moire and angle

This is an output of wanniertools (full tb solution)
![nmoire](figure/nmoire_angle.png)

## Glist Construction

for tight binding (nsymm case, specific valley 1)

![](figure/glist_tb_v_1.png)


## Example Output

### Tight Binding

For `n_moire = 30`, four bands

![eg1](figure/tb_n30_2band.png)

### Continuum

For `n_moire = 31`, four bands:

![eg3](figure/continuum_2band.png)

For `n_moire = 31`, more bands:

![eg4](figure/continuum.png)
