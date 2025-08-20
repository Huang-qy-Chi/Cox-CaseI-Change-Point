It works for the fixed change-point Cox model $$\Lambda(t|Z, Z_2)=\Lambda_0(t)\exp(Z^\top\beta+\tilde{Z}^\top\gamma I(Z_2>\zeta))$$ 
under current status data (case-I interval censored data), where $t\in [0, M]$, $Z\in\mathbb{R}^p$, $\tilde{Z}=(1, Z^\top)^\top$, 
$Z_2\in\mathbb{R}$, $\Lambda_0(\cdot)$ denotes the baseline cumulative hazard and $I(\cdot)$ represents the indicator. 

I recommend considering the Python version first, as the R version was uploaded last year and may cause confusion due to its very low quality. 
Some $\textbf{VERY BAD}$ things happened when I wrote code in ``R_version" during 2024, which made me unwilling to update them, and now I have finally decided to rewrite it with Python. 
Actually, this outdated technique should be eliminated. These files are just kept as a souvenir.

The code in file ``Python_version" recommends Python version>=3.13.2, Numpy >=2.2.4, and Scipy>=1.15.2.

Copyright © 2025 Q. Huang. All rights reserved.

20/08/2025, Hung Hom, Kowloon, Hong Kong, China.
