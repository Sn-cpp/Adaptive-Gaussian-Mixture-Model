**Initialization**

Given a Gaussian Mixture Model intitalized with a RGB frame of size (H, W, 3) and K components, the internal attributes are:
+ `means` ($\mu$) of size (H, W, K, 3): contains means of the Gaussian distributions $k$ ($0 \leq k < K$) of each pixel. The $k_0$ component is intitalized using the first frame, i.e: `means[:, :, 0, :] = first_frame`.
+ `vars`  ($\sigma^2$) of size (H, W, K) contains variances of the Gaussian distributions $k$ ($0 \leq k < K$) of each pixel, filled with large constant value (400).
+ `weights` of size (H, W, K) contains weights of components of each pixel. The weights are initialized with $1.0$ for the first component (i.e: `self.weights[:, :, 0] = 1.0`) and $0.0$ for the rest.

**Update phase**

For each incoming frame `frame`:

+ For each pixel at `(y, x)`:
    
    + Compute the Euler distance $D$ between `frame[y, x]` and $\mu_k$, for each $0 \leq k < K$.

    + Let `matched` (K,) be a binary mask, compare the Euler distance $D_k$ with the corresponding $\sigma_k^2$ and a given threshold $\alpha$:
    
        `matched[k]` = $\begin{cases}
            \text{True if } D_k < \alpha\cdot\sigma_k\\
            \text{False, otherwise}  
        \end{cases}$

    + Select `k_min` where `matched[k_min]` = `True` and $D_{\text{k\_min}} = min(D)$.

    + Decay weights: `weights` = $(1.0-\alpha) \cdot$`weights`
    
    + If `k_min` exists:

        + `weights[k_min]` += $\alpha$

        + Update mean: $\mu_\text{k\_min}$ = $(1.0-\alpha) \cdot \mu_\text{k\_min}$ + $\alpha \cdot$ `frame[y, x]`

        + Update variance: $\sigma^2_\text{k\_min}$ = $(1.0-\alpha) \cdot \sigma_\text{k\_min}^2$ + $\alpha \cdot D_\text{k\_min}^2$

    + For all unmatched components $j$ where `matched[j]` = `False`:

        + Select `weakest` where `matched[weakest]` = `False` and `weights[weakest]` = $min(\text{weights})$ 

        + Replace mean: $\mu_\text{weakest}$ = `frame[y, x]`
        
        + Replace variance: $\sigma_\text{weakest}^2 = 400.0$
        
        + Replace weight: `weights[weakest]` = $0.01$
    
    + Normalize the `weights` so that they sum to $1.0$.

**Predict**

For each incoming frame `frame`:

+ For each pixel at `y, x`:

    + Compute the Euler distance $D$ between `frame[y, x]` and $\mu_k$, for each $0 \leq k < K$.

    + Let `matched` (K,) be a binary mask, compare the Euler distance $D_k$ with the corresponding $\sigma_k^2$ and a given threshold $\alpha$:

        `matched[k]` = $\begin{cases}
            \text{True if } D_k < \alpha\cdot\sigma_k\\
            \text{False, otherwise}  
        \end{cases}$

    + Sort weights and `matched` by $\frac{\textbf{weights}}{\sigma^2}$

    + Compute cumulative sum of weights of sorted components, denoted `cumulative_weights` 

    + Select `background_components` where `matched` = `True` and  `cumulative_weights` < `background_threshold` (always include the first component).

    

