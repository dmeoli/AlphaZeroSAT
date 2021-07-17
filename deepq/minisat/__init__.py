from gym.envs.registration import register

from .minisat.gym.MiniSATEnv import (gym_sat_Env, gym_sat_sort_Env, gym_sat_permute_Env,
                                     gym_sat_graph_Env, gym_sat_graph2_Env)

# sat solver envs
register(id='gym_sat_Env-v0',
         entry_point='gym.envs.SatSolver:gym_sat_Env')

# sat solver envs that always sort the rows of states
register(id='gym_sat_Env-v1',
         entry_point='gym.envs.SatSolver:gym_sat_sort_Env')

# sat solver envs that always permute the input file
register(id='gym_sat_Env-v2',
         entry_point='gym.envs.SatSolver:gym_sat_permute_Env')

# sat solver envs for graph coloring (no preprocess or post process)
register(id='gym_sat_Env-v3',
         entry_point='gym.envs.SatSolver:gym_sat_graph_Env')

# sat solver envs for graph coloring (no preprocess or post process, larger)
register(id='gym_sat_Env-v4',
         entry_point='gym.envs.SatSolver:gym_sat_graph2_Env')
