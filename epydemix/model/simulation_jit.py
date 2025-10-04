"""
Numba JIT-accelerated simulation engine for epydemix.

This module provides optimized versions of the core stochastic simulation loop
using numba's just-in-time compilation for significant performance improvements.
"""

import numpy as np
from numba import njit
from typing import Dict, List, Tuple, Optional, Any
import copy
from .numba_random import multinomial_sequential


def prepare_simulation_data(epimodel, contact_matrices: List[Dict], parameters: Dict, dt: float) -> Dict:
    """
    Prepare model data for JIT-compiled simulation.
    
    Converts dynamic EpiModel structures into static numpy arrays that numba can compile.
    
    Args:
        epimodel: The EpiModel instance
        contact_matrices: List of contact matrix dictionaries for each time step
        parameters: Parameter definitions dictionary
        dt: Time step size
        
    Returns:
        Dictionary containing:
            - contact_matrix_array: (T, N, N) array of contact matrices
            - transition_specs: List of transition specifications
            - compartment_map: Dictionary mapping compartment names to indices
            - n_transitions: Number of transitions
            - pop_sizes: Population sizes array
    """
    T = len(contact_matrices)
    N = len(epimodel.population.Nk)
    C = len(epimodel.compartments)
    
    # Stack contact matrices into 3D array (T, N, N)
    contact_matrix_array = np.zeros((T, N, N), dtype=np.float64)
    for t, cm_dict in enumerate(contact_matrices):
        contact_matrix_array[t] = cm_dict["overall"]
    
    # Build transition specifications as structured data
    transition_specs = []
    for comp in epimodel.compartments:
        transitions = epimodel.transitions[comp]
        for tr in transitions:
            source_idx = epimodel.compartments_idx[tr.source]
            target_idx = epimodel.compartments_idx[tr.target]
            
            # Extract parameters based on transition kind
            if tr.kind == "spontaneous":
                # Spontaneous: single parameter (rate)
                if isinstance(tr.params, str):
                    # Evaluate string expression
                    env_copy = copy.deepcopy(parameters)
                    rate_array = _evaluate_parameter(tr.params, env_copy, T, N)
                else:
                    rate_array = np.full((T, N), tr.params, dtype=np.float64)
                
                spec = {
                    'kind': 0,  # 0 = spontaneous
                    'source_idx': source_idx,
                    'target_idx': target_idx,
                    'rate': rate_array,
                    'agent_idx': -1,
                    'transition_idx': epimodel.transitions_idx[f"{tr.source}_to_{tr.target}"]
                }
                
            elif tr.kind == "mediated":
                # Mediated: (rate, agent_compartment)
                rate_param = tr.params[0]
                agent_name = tr.params[1]
                agent_idx = epimodel.compartments_idx[agent_name]
                
                if isinstance(rate_param, str):
                    env_copy = copy.deepcopy(parameters)
                    rate_array = _evaluate_parameter(rate_param, env_copy, T, N)
                else:
                    rate_array = np.full((T, N), rate_param, dtype=np.float64)
                
                spec = {
                    'kind': 1,  # 1 = mediated
                    'source_idx': source_idx,
                    'target_idx': target_idx,
                    'rate': rate_array,
                    'agent_idx': agent_idx,
                    'transition_idx': epimodel.transitions_idx[f"{tr.source}_to_{tr.target}"]
                }
            else:
                raise ValueError(f"Unknown transition kind: {tr.kind}")
            
            transition_specs.append(spec)
    
    return {
        'contact_matrix_array': contact_matrix_array,
        'transition_specs': transition_specs,
        'pop_sizes': epimodel.population.Nk.astype(np.float64),
        'dt': dt,
        'n_compartments': C,
        'n_demographics': N,
        'n_transitions': epimodel.n_transitions
    }


def _evaluate_parameter(expr: str, env: dict, T: int, N: int) -> np.ndarray:
    """
    Evaluate a parameter expression over time.
    
    Args:
        expr: String expression
        env: Environment dictionary
        T: Number of time steps
        N: Number of demographic groups
        
    Returns:
        Array of shape (T, N) with evaluated parameter values
    """
    from ..utils.utils import evaluate
    
    result = evaluate(expr=expr, env=env)
    if result.ndim == 1:
        # Time-varying only, expand to (T, N)
        return np.tile(result[:T, np.newaxis], (1, N))
    elif result.ndim == 2:
        # Already (T, N)
        return result[:T, :]
    else:
        raise ValueError(f"Unexpected parameter shape: {result.shape}")


@njit(cache=True)
def _compute_spontaneous_transition_prob_vectorized(rate: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute spontaneous transition probability for all demographic groups.
    
    Args:
        rate: Array of shape (N,) with rates for each demographic group
        dt: Time step size
        
    Returns:
        Array of shape (N,) with transition probabilities
    """
    return 1.0 - np.exp(-rate * dt)


@njit(cache=True)
def _compute_mediated_transition_prob_vectorized(rate: np.ndarray, agent_pop: np.ndarray, 
                                                  contact_matrix: np.ndarray, pop_sizes: np.ndarray, 
                                                  dt: float) -> np.ndarray:
    """
    Compute mediated transition probability for all demographic groups.
    
    Args:
        rate: Array of shape (N,) with rates for each demographic group
        agent_pop: Array of shape (N,) with agent compartment populations
        contact_matrix: Array of shape (N, N) contact matrix
        pop_sizes: Array of shape (N,) with population sizes
        dt: Time step size
        
    Returns:
        Array of shape (N,) with transition probabilities
    """
    # Vectorized contact matrix computation - single operation for all demographics
    interaction = np.sum(contact_matrix * agent_pop / pop_sizes, axis=1)
    return 1.0 - np.exp(-rate * interaction * dt)


@njit(cache=True)
def _stochastic_simulation_jit(
    T: int,
    N: int,
    C: int,
    n_transitions: int,
    contact_matrix_array: np.ndarray,
    transition_kinds: np.ndarray,
    transition_sources: np.ndarray,
    transition_targets: np.ndarray,
    transition_rates: np.ndarray,
    transition_agents: np.ndarray,
    transition_indices: np.ndarray,
    pop_sizes: np.ndarray,
    initial_conditions: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled core stochastic simulation loop with vectorized operations.
    
    This version maintains vectorized numpy operations across demographic groups
    while benefiting from JIT compilation of the outer loops.
    """
    # Pre-allocate arrays
    compartments_evolution = np.zeros((T + 1, C, N), dtype=np.float64)
    transitions_evolution = np.zeros((T, n_transitions, N), dtype=np.float64)
    compartments_evolution[0] = initial_conditions
    
    # Pre-allocate working arrays
    prob = np.zeros((C, N), dtype=np.float64)
    new_pop = np.zeros((C, N), dtype=np.float64)
    
    # Build compartment -> transitions mapping
    comp_to_trans_count = np.zeros(C, dtype=np.int32)
    for i in range(len(transition_sources)):
        comp_to_trans_count[transition_sources[i]] += 1
    
    # Pre-allocate proper size for each compartment's transition list
    max_trans = np.max(comp_to_trans_count)
    comp_to_trans = np.full((C, max_trans), -1, dtype=np.int32)
    comp_trans_idx = np.zeros(C, dtype=np.int32)
    
    for i in range(len(transition_sources)):
        source = transition_sources[i]
        idx = comp_trans_idx[source]
        comp_to_trans[source, idx] = i
        comp_trans_idx[source] += 1
    
    # Main simulation loop
    for t in range(T):
        new_pop[:, :] = compartments_evolution[t]
        
        # Loop over compartments
        for comp_idx in range(C):
            n_trans_comp = comp_to_trans_count[comp_idx]
            
            if n_trans_comp == 0:
                continue
            
            # Reset probability array
            prob[:, :] = 0.0
            
            # Get current population for this compartment (vectorized across demographics)
            current_pop = compartments_evolution[t, comp_idx, :]
            
            # Check if compartment has any population
            if not np.any(current_pop > 0):
                continue
            
            # Compute transition probabilities (vectorized across demographics)
            for i in range(n_trans_comp):
                trans_idx = comp_to_trans[comp_idx, i]
                if trans_idx == -1:
                    break
                    
                target = transition_targets[trans_idx]
                kind = transition_kinds[trans_idx]
                rate = transition_rates[trans_idx, t, :]  # Vector of rates for all demographics
                
                if kind == 0:  # Spontaneous - vectorized
                    prob[target, :] += _compute_spontaneous_transition_prob_vectorized(rate, dt)
                    
                elif kind == 1:  # Mediated - vectorized
                    agent_idx = transition_agents[trans_idx]
                    agent_pop = compartments_evolution[t, agent_idx, :]
                    prob[target, :] += _compute_mediated_transition_prob_vectorized(
                        rate, agent_pop, contact_matrix_array[t], pop_sizes, dt
                    )
            
            # Compute remaining probability (vectorized)
            prob[comp_idx, :] = 1.0 - np.sum(prob, axis=0)
            
            # Sample transitions for each demographic group
            # This loop is necessary as multinomial must be called per demographic group
            for n in range(N):
                if current_pop[n] > 0:
                    # Extract probability vector for this demographic
                    p_vec = prob[:, n].copy()
                    
                    # Normalize to handle numerical errors
                    p_sum = np.sum(p_vec)
                    if p_sum > 0:
                        p_vec = p_vec / p_sum
                    else:
                        p_vec[comp_idx] = 1.0
                    
                    # Sample from multinomial using our optimized numba implementation
                    delta = multinomial_sequential(int(current_pop[n]), p_vec)
                    
                    # Record transitions - only for actual transitions (not staying)
                    for i in range(n_trans_comp):
                        trans_idx = comp_to_trans[comp_idx, i]
                        if trans_idx == -1:
                            break
                        target = transition_targets[trans_idx]
                        tr_output_idx = transition_indices[trans_idx]
                        transitions_evolution[t, tr_output_idx, n] += delta[target]
                    
                    # Update populations
                    delta_sum = np.sum(delta)
                    new_pop[comp_idx, n] -= delta_sum
                    for c in range(C):
                        new_pop[c, n] += delta[c]
        
        compartments_evolution[t + 1] = new_pop
    
    return compartments_evolution[1:], transitions_evolution


def stochastic_simulation_jit(
    T: int,
    contact_matrices: List[Dict[str, np.ndarray]],
    epimodel,
    parameters: Dict,
    initial_conditions: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-accelerated stochastic simulation.
    
    This is the main entry point that wraps the numba-compiled simulation loop.
    
    Args:
        T: Number of time steps
        contact_matrices: List of contact matrix dictionaries
        epimodel: EpiModel instance
        parameters: Model parameters
        initial_conditions: Initial population distribution
        dt: Time step size
        
    Returns:
        compartments_evolution: (T, C, N) array of compartment populations
        transitions_evolution: (T, n_transitions, N) array of transition counts
    """
    # Prepare data for JIT compilation
    prepared = prepare_simulation_data(epimodel, contact_matrices, parameters, dt)
    
    # Extract transition data into arrays
    n_trans = len(prepared['transition_specs'])
    N = prepared['n_demographics']
    C = prepared['n_compartments']
    
    transition_kinds = np.zeros(n_trans, dtype=np.int32)
    transition_sources = np.zeros(n_trans, dtype=np.int32)
    transition_targets = np.zeros(n_trans, dtype=np.int32)
    transition_rates = np.zeros((n_trans, T, N), dtype=np.float64)
    transition_agents = np.zeros(n_trans, dtype=np.int32)
    transition_indices = np.zeros(n_trans, dtype=np.int32)
    
    for i, spec in enumerate(prepared['transition_specs']):
        transition_kinds[i] = spec['kind']
        transition_sources[i] = spec['source_idx']
        transition_targets[i] = spec['target_idx']
        transition_agents[i] = spec['agent_idx']
        transition_indices[i] = spec['transition_idx']
        transition_rates[i] = spec['rate']
    
    # Call JIT-compiled simulation
    compartments_evolution, transitions_evolution = _stochastic_simulation_jit(
        T=T,
        N=N,
        C=C,
        n_transitions=prepared['n_transitions'],
        contact_matrix_array=prepared['contact_matrix_array'],
        transition_kinds=transition_kinds,
        transition_sources=transition_sources,
        transition_targets=transition_targets,
        transition_rates=transition_rates,
        transition_agents=transition_agents,
        transition_indices=transition_indices,
        pop_sizes=prepared['pop_sizes'],
        initial_conditions=initial_conditions,
        dt=dt
    )
    
    return compartments_evolution, transitions_evolution
