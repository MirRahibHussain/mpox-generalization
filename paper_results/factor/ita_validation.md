# ITA proxy check on Fitzpatrick17k (atlas pool)

Images with both an FST label and an ITA estimate: 15902 of 16557 (ITA could not be estimated for 119; FST missing for 562).

Spearman rho(ITA, FST) = -0.487 (95% bootstrap CI -0.501 to -0.474; p = 0).

## ITA by Fitzpatrick skin type
| FST | n | ITA Q1 | ITA median | ITA Q3 |
| --- | --- | --- | --- | --- |
| 1 | 2907 | 32.4 | 55.9 | 69.0 |
| 2 | 4773 | 12.5 | 35.8 | 54.1 |
| 3 | 3299 | 7.6 | 24.6 | 38.4 |
| 4 | 2777 | -3.9 | 13.0 | 26.2 |
| 5 | 1523 | -24.3 | -2.5 | 15.4 |
| 6 | 623 | -56.5 | -35.5 | -1.0 |

## ITA band vs FST group (rows: FST 1-2 / 3-4 / 5-6; columns: ITA band)
| FST group | light | intermediate | dark |
| --- | --- | --- | --- |
| light | 4010 | 2143 | 1527 |
| intermediate | 946 | 3011 | 2119 |
| dark | 151 | 460 | 1535 |

Agreement 0.538; linearly weighted kappa 0.345; light/dark swapped in 0.106 of images.

## ITA-band mix: mpox images vs expected atlas draw (all mpox images of the dataset; training splits are subsets)
| dataset | draw | light | intermediate | dark | unknown |
| --- | --- | --- | --- | --- | --- |
| msid | mpox images (target) | 0.318 | 0.242 | 0.419 | 0.022 |
| msid | atlas, cond | 0.34 | 0.338 | 0.315 | 0.008 |
| msid | atlas, cond_tone | 0.318 | 0.242 | 0.419 | 0.022 |
| msld_v2 | mpox images (target) | 0.335 | 0.246 | 0.419 | 0.0 |
| msld_v2 | atlas, cond | 0.34 | 0.338 | 0.315 | 0.008 |
| msld_v2 | atlas, cond_tone | 0.335 | 0.246 | 0.419 | 0.0 |

