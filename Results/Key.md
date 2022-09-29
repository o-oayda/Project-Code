# Key
This document describes the different parameters selected for each data trial.

- 1-08-r1 to r3: 10% uncertainty, `obsPolar = (0.7, 4)`, `nside = 16`.
- 4-08 as above.
- 4-08-r2 with `fastRotateMatrix` I'm pretty sure
- 5-08-r2 with `quaternionRotate`.
- 5-08-r3 back to `fastRotateMatrix`.
- to r6 as above
- 7-08 `obsPolar = (np.pi/2, 400)`, 10% uncertainty
- 7-08-r6 up to 8-08-r4 30% uncertainty, `obsPolar = (0.7,4)`

## Results for 8-08
Results Table 1: 10%, (0.7,4)
Results Table 2: 10%, (pi/2,4)
Results Table 3: 30%, (0.7,4)

## New
- 14-08: 1%, (0.7,4).
- 15-08: 1%, 10%, 30%, 50% and (np.pi - 0.7, 4 - np.pi)
- 22-08: 



Note:
- `nside = 16`, `obsSpeed = 0.001` and priors are uniform across all angles of a sphere with $v \sim U(0,0.01)$ for all tests below:
- From the second trial in the 27-09 for Test A, `NestedSampler` was used as oppossed to `DynamicNestedSampler` on the basis that the former is better for evaluating the evidence (and faster). This has produced better results.

| Test | Characteristics | Files |
|------|---|---|
|Test A| $\sigma = 10\%$, `obsPolar = (0.7,4)` | |
|Test B| $\sigma = 10\%$, `obsPolar = (np.pi/2,4)`| |
|Test C| $\sigma = 30\%$, `obsPolar = (0.7,4)`| |
|Test D| $\sigma = 1\%$, `obsPolar = (np.pi - 0.7, 4 - np.pi)`| |
|Test E| $\sigma = 10\%$, `obsPolar = (np.pi - 0.7, 4 - np.pi)`| |
|Test F| $\sigma = 30\%$, `obsPolar = (np.pi - 0.7, 4 - np.pi)`| |
|Test G| $\sigma = 50\%$, `obsPolar = (np.pi - 0.7, 4 - np.pi)`| |
|Test H| $\sigma = 10\%$, `obsPolar = (np.pi/2 - 30*np.pi/180, np.pi/2)`| |
|Test I| $\sigma = 30\%$, `obsPolar = (np.pi/2 - 30*np.pi/180, np.pi/2)`| |
|Test J| $\sigma = 50\%$, `obsPolar = (np.pi/2 - 30*np.pi/180, np.pi/2)`| |

Note: results1 = tableA; results2 = tableB; results3 = tableC