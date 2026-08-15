# Results Summary — House Price Regression

**Dataset:** `House_price.csv` (999 rows, no missing values, no duplicates)

**Features used:** Avg. Area Income, House Age, Number of Rooms, Number of Bedrooms, Area Population
**Target:** Price

**Split:** 75% train / 25% test (random_state=1)

## Test-set performance
| Metric | Value |
|---|---|
| RMSE | $96,103 |
| R² | 0.927 |

## Coefficients
| Feature | Coefficient |
|---|---|
| Avg. Area Income | 21.39 |
| House Age | 165,721.82 |
| Number of Rooms | 120,140.70 |
| Number of Bedrooms | -3,922.28 |
| Area Population | 14.74 |
| Intercept | -2,578,730.00 |

## Interpretation

The model explains about 93% of the variance in house price on unseen data, with a typical prediction error of roughly $96k against an average price of $1.25M (~7-8% of the mean). Income, house age, and room count all have positive coefficients that match intuition — richer areas, older (more established) houses, and more rooms are priced higher. Number of bedrooms comes out slightly negative, which looks odd, but it's a multicollinearity artifact: bedrooms and rooms are correlated (~0.7), so once "Number of Rooms" is in the model it absorbs most of that signal and bedrooms picks up a small, near-zero adjustment rather than a real negative effect. Overall this is a solid, well-behaved linear model — good enough to use for ballpark price estimates, though a next step would be to check for interaction effects (e.g. income × population) or try dropping bedrooms to see if the test R² holds.
