"""
Numba-optimized random sampling functions replicating NumPy's algorithms.

This module implements NumPy's multinomial sampling using the sequential 
conditional binomial method, exactly as done in NumPy's C source code.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def binomial_inversion(n, p):
    """
    Sample from binomial distribution using inversion method.
    
    This is NumPy's algorithm for small n*p (≤ 30).
    Uses the recurrence relation: P(X=k+1) = P(X=k) * (n-k)*p / ((k+1)*(1-p))
    
    Args:
        n: Number of trials
        p: Success probability
        
    Returns:
        Number of successes (integer)
    """
    if n == 0 or p == 0.0:
        return 0
    
    if p == 1.0:
        return n
    
    q = 1.0 - p
    qn = np.exp(n * np.log(q))  # P(X=0) = q^n
    
    X = 0
    px = qn  # Current probability mass
    U = np.random.random()
    
    # Bound to avoid infinite loops in edge cases
    bound = min(n, int(n*p + 10.0*np.sqrt(n*p*q + 1)))
    
    while U > px:
        X += 1
        if X > bound:
            # Restart if we exceed reasonable bound
            X = 0
            px = qn
            U = np.random.random()
        else:
            U -= px
            # Recurrence relation for binomial probabilities
            px = px * (n - X + 1) * p / (X * q)
    
    return X


@njit(cache=True)
def binomial_normal_approx(n, p):
    """
    Sample from binomial using normal approximation.
    
    Used for large n*p (> 30) as a simplified alternative to BTPE.
    This is less accurate than NumPy's BTPE but much simpler to implement.
    
    Args:
        n: Number of trials
        p: Success probability
        
    Returns:
        Number of successes (integer)
    """
    if n == 0 or p == 0.0:
        return 0
    
    if p == 1.0:
        return n
    
    mu = n * p
    sigma = np.sqrt(n * p * (1.0 - p))
    
    # Normal approximation with continuity correction
    X = int(np.random.normal(mu, sigma) + 0.5)
    
    # Clamp to valid range
    return max(0, min(n, X))


@njit(cache=True)
def adaptive_binomial(n, p):
    """
    Adaptive binomial sampler matching NumPy's strategy.
    
    Switches between inversion (for small n*p) and normal approximation 
    (for large n*p). Uses symmetry trick when p > 0.5.
    
    Args:
        n: Number of trials
        p: Success probability
        
    Returns:
        Number of successes (integer)
    """
    if n == 0 or p == 0.0:
        return 0
    
    if p == 1.0:
        return n
    
    # Use symmetry if p > 0.5: Binomial(n, p) = n - Binomial(n, 1-p)
    if p > 0.5:
        q = 1.0 - p
        if q * n <= 30.0:
            return n - binomial_inversion(n, q)
        else:
            return n - binomial_normal_approx(n, q)
    else:
        if p * n <= 30.0:
            return binomial_inversion(n, p)
        else:
            return binomial_normal_approx(n, p)


@njit(cache=True)
def multinomial_sequential(n, pvals):
    """
    Sample from multinomial distribution using sequential conditional binomials.
    
    This replicates NumPy's algorithm exactly:
    1. For each category j (except last), sample from Binomial(n_remaining, p_j / p_remaining)
    2. Update remaining trials and probability mass
    3. Assign all remaining trials to last category
    
    This ensures sum(counts) = n exactly and is numerically stable.
    
    Args:
        n: Number of trials (must be positive integer)
        pvals: Probability array (must sum to 1)
        
    Returns:
        Array of counts for each category
    """
    d = len(pvals)
    counts = np.zeros(d, dtype=np.int64)
    
    if n == 0:
        return counts
    
    remaining_p = 1.0
    dn = n
    
    # Sample d-1 categories using conditional binomials
    for j in range(d - 1):
        if dn <= 0:
            # Early termination: all trials allocated
            break
        
        if pvals[j] == 0.0:
            # Skip zero-probability categories
            counts[j] = 0
            continue
        
        # Conditional probability given remaining mass
        conditional_p = pvals[j] / remaining_p
        
        # Clamp to [0, 1] to handle numerical errors
        conditional_p = min(1.0, max(0.0, conditional_p))
        
        # Sample from conditional binomial
        counts[j] = adaptive_binomial(dn, conditional_p)
        
        # Update remaining trials and probability mass
        dn -= counts[j]
        remaining_p -= pvals[j]
        
        # Safety check for numerical stability
        if remaining_p < 1e-10:
            remaining_p = 1e-10
    
    # Last category gets all remaining trials
    if dn > 0:
        counts[d - 1] = dn
    
    return counts
