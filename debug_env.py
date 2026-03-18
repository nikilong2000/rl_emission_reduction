import sys
sys.path.append('.')
from ppo.train_ppo import EmissionControlEnv

env = EmissionControlEnv(dataset_path="/Users/niklaslongschiefelbein/local_documents/THESIS/code/rl_emission_reduction/02_rl_control/data_train/WLTC.csv")
obs, info = env.reset()
print("step 0 info:", info)
for i in range(15):
    obs, reward, done, trunc, info = env.step(env.action_space.sample())
    t_speed = info.get("target_speed", None)
    if t_speed is not None and t_speed > 0:
         print(f"Target > 0 at step {i}: {t_speed}")

print("info at step 15:", info)
# skip ahead to where WLTC speed shouldn't be 0
for i in range(16, 50):
    obs, reward, done, trunc, info = env.step(env.action_space.sample())
    t_speed = info.get("target_speed", None)
    print(f"Step {i} speed: {t_speed}")
