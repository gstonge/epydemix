"""Quick diagnostic script to measure where time is spent in JIT simulation."""

import numpy as np
import time
from epydemix import EpiModel

# Create population model
model = EpiModel(
    name="Diagnostic",
    compartments=["S", "E", "I", "R"],
    parameters={"beta": 0.5, "sigma": 0.2, "gamma": 0.1}
)
model.import_epydemix_population(
    population_name="United_States_California",
    contact_layers=["all"]
)
model.add_transition(source="S", target="E", kind="mediated", params=("beta", "I"))
model.add_transition(source="E", target="I", kind="spontaneous", params="sigma")
model.add_transition(source="I", target="R", kind="spontaneous", params="gamma")

print("Running diagnostic on JIT simulation...")
print(f"Population: {model.population.total_population:,.0f}")
print(f"Age groups: {model.population.num_groups}")
print()

# Patch the stochastic_simulation_jit to add timing
from epydemix.model import simulation_jit
original_sim_jit = simulation_jit.stochastic_simulation_jit

def timed_sim_jit(T, contact_matrices, epimodel, parameters, initial_conditions, dt):
    t0 = time.time()
    prepared = simulation_jit.prepare_simulation_data(epimodel, contact_matrices, parameters, dt)
    prep_time = time.time() - t0
    
    t1 = time.time()
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
    
    extract_time = time.time() - t1
    
    t2 = time.time()
    result = simulation_jit._stochastic_simulation_jit(
        T=T, N=N, C=C,
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
    jit_time = time.time() - t2
    
    print(f"  Preparation: {prep_time*1000:.2f}ms")
    print(f"  Extraction:  {extract_time*1000:.2f}ms")
    print(f"  JIT execution: {jit_time*1000:.2f}ms")
    print(f"  TOTAL: {(prep_time + extract_time + jit_time)*1000:.2f}ms")
    
    return result

simulation_jit.stochastic_simulation_jit = timed_sim_jit

# Run single simulation with JIT
print("=== First JIT run (includes compilation) ===")
results_jit = model.run_simulations(
    start_date="2020-01-01",
    end_date="2020-12-31",
    Nsim=1,
    use_jit=True
)

print()
print("=== Second JIT run (no compilation) ===")
results_jit = model.run_simulations(
    start_date="2020-01-01",
    end_date="2020-12-31",
    Nsim=1,
    use_jit=True
)

print()
print("=== Original version for comparison ===")
t0 = time.time()
results_orig = model.run_simulations(
    start_date="2020-01-01",
    end_date="2020-12-31",
    Nsim=1,
    use_jit=False
)
orig_time = time.time() - t0
print(f"  Original total: {orig_time*1000:.2f}ms")
