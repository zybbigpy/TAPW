# Tight Binding Method for Twist Bilayer Graphene

## Graphene Coordinates

See the notes.

## Code Structures

- [x] Tight Binding Model (CPU)

- [x] Tight Binding Model in Magnetic Field

## Dependent Module

1. Numpy
2. Scipy

## Relationship between n_moire and angle

This is an output of wanniertools (full tb solution)
![nmoire](fig/nmoire_angle.png)

## Example Output

For `n_moire = 30, valley = 1`:

![eg1](fig/band_n_30_v_1.png)

For `n_moire = 30, valley = -1`:

![eg2](fig/band_n_30_v_-1.png)