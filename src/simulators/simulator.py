class Simulator:
    def __init__(self, dt=0.1):
        self.dt = dt

    def run(self, num_steps, filtering, bounding_method, experiment_number):
        exp = self.experiments.get(experiment_number)
        if exp is None:
            raise ValueError(f"Unknown experiment number: {experiment_number}")
        return exp["run"](num_steps, filtering, bounding_method)

    def plot(self):
        # Implement the plotting logic here
        pass
