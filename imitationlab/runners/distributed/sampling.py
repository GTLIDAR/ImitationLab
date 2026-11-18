"""Pool per-step adaptive reset statistics without multiplying shared priors."""

import torch
import torch.distributed as dist


class ResetCountSynchronizer:
    def __init__(self, sampler, device):
        self.sampler = sampler
        self.device = device
        self.previous = torch.empty((2, sampler.num_visits.numel()), device=device)

    def begin_step(self):
        self.previous[0].copy_(self.sampler.num_visits)
        self.previous[1].copy_(self.sampler.num_failures)

    def finish_step(self):
        current = torch.stack((self.sampler.num_visits, self.sampler.num_failures)).to(
            self.device
        )
        delta = current - self.previous
        dist.all_reduce(delta)
        current = self.previous + delta
        self.sampler.num_visits.copy_(current[0])
        self.sampler.num_failures.copy_(current[1])
