from typing import List, Optional, Tuple, Union
from . import privacy_analysis

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


class RDPAccountant:
    def __init__(self):
        self.steps = []

    def step(self, noise_multiplier: float, sample_rate: float):
        self.steps.append((noise_multiplier, sample_rate))

    def get_privacy_spent(
        self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        if alphas is None:
            alphas = DEFAULT_ALPHAS
        rdp = sum(
            [
                privacy_analysis.compute_rdp(sample_rate, noise_multiplier, 1, alphas)
                for (sample_rate, noise_multiplier) in self.steps
            ]
        )

        eps, best_alpha = privacy_analysis.get_privacy_spent(alphas, rdp, delta)

        return float(eps), float(best_alpha)


if __name__ == "__main__":
    accountant = RDPAccountant()
    accountant.step(1.0, 1e-3)

    print(accountant.get_privacy_spent(delta=1e-5))
