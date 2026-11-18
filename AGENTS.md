# Working in ImitationLab

Start with README.md and the relevant setup, training, or contribution guide.
Run commands from the repository root. Pixi manages the native packages and
Isaac Lab. The optional official SuperMimic baseline uses its own uv environment.

Keep policy learners and encoders simulator-independent where possible. Put the
G1 observation, action, reward, and reset contract in the environment package.
Keep live and imagined observation assembly consistent. `imitation-g1-v1` is
the only task; preserve its checkpoint compatibility. Use a new public task
version when the MDP or policy input contract changes.

Run the affected tests first. Use `pixi run test-default` for the CPU suites,
`pixi run -e static ci-static` for static checks, and the documented local GPU
smoke when changing simulator or trainer wiring. Use only local GPUs unless
the user explicitly authorizes work on a specific remote machine.

Keep data and run artifacts outside Git. Record the resolved task, interface,
learner, data identity, encoder weights, and seed when reporting results.
