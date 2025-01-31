import numpy as np
from mesa import Model

from agents import GenCoAgent


class ElectricityMarket(Model):
    """Model of electricity market auctions with different pricing mechanisms"""

    def __init__(self, **kwargs) -> None:
        """Initialize the electricity market model.

        Args:
            **kwargs: Model parameters including:
                steps (int): Number of steps to run (default: 1000)
                n (int): Number of generating companies (default: 5)
                m (float): Slope of price-capacity relationship (default: -5)
                b (float): Intercept of price-capacity relationship (default: 1000)
                min_cost (float): Minimum generation cost (default: 20)
                max_cost (float): Maximum generation cost (default: 200)
                min_error (float): Minimum error in cost expectations (default: 1)
                max_error (float): Maximum error in cost expectations (default: 20)
                min_padding (float): Minimum padding factor for bids (default: 50)
                max_padding (float): Maximum padding factor for bids (default: 99)
                cap_multiplier (float): Multiplier for price cap (default: 1.2)
        """
        super().__init__()

        self.mechanism = "uniform"  # leave slot for mechanism

        # Set defaults
        defaults = {
            "steps": 1000,
            "n": 5,
            "m": -5,
            "b": 1000,
            "min_cost": 20,
            "max_cost": 200,
            "min_error": 1,
            "max_error": 20,
            "min_padding": 50,
            "max_padding": 99,
            "cap_multiplier": 1.2,
        }

        # Update defaults with provided kwargs
        defaults.update(kwargs)

        # Store parameters as instance variables
        for key, value in defaults.items():
            setattr(self, key, value)

        self.rng = np.random.default_rng()  ## set up random number generator

        # Generate costs and capacities
        self.costs = self.rng.uniform(self.min_cost, self.max_cost, self.n)
        self.capacities = self.m * self.costs + self.b
        self.price_cap = max(self.costs) * self.cap_multiplier

        # Generate agent expectations
        errors = (
            self.rng.uniform(self.min_error, self.max_error, self.n * self.n)
            / 100
            * np.random.choice([-1, 1], size=self.n * self.n)
        ).reshape([self.n, self.n])
        paddings = self.rng.uniform(self.min_padding, self.max_padding, self.n) / 100

        # Create agents
        for i_n in range(self.n):
            agent = GenCoAgent(
                i_n,
                self,
                self.costs[i_n],
                self.capacities[i_n],
                errors[i_n],
                paddings[i_n],
            )
            self.register_agent(agent)

        self.running = True
        self.datacollector = None  # We can add this later for data collection

        # Add results tracking
        self.RTO_costs = np.zeros(self.steps)  # Store RTO costs for each step
        self.GenCo_profits = np.zeros(
            (self.steps, self.n)
        )  # Store profits for each GenCo at each step
        self.step_count = 0

    def find_capacities(self, prices, m, b):
        # precisely the function from the source notebook
        return m * prices + b

    def create_expectations(
        self, n: int, capacities, costs
    ):
        # precisely the function from the source notebook
        capacity_expectations = np.round(capacities, -2)
        cost_expectations = np.empty([n, n])
        expected_rankings = np.empty([n, n], int)
        for g in range(n):
            cost_expectations[g] = (1 + self.errors[g]) * costs
            expected_rankings[g] = np.argsort(cost_expectations[g])
        return (capacity_expectations, cost_expectations, expected_rankings)

    def step(self):
        """Advance model by one step"""
        # telemetry back to the caller:
        if self.step_count % 100 == 0:
            print(f"Time: {self.step_count}")

        # commented code from the source notebook will have two hashes, like the following:
        ##      N = rng.uniform(int(np.mean(capacities)), int(np.sum(capacities)))
        N = self.rng.uniform(  # initialized in the constructor from np.random
            int(np.mean(self.capacities)), int(np.sum(self.capacities))
        )

        # the agent generate bids line:
        ##      bids = generate_bids(n, N, mechanism, capacity_expectations, 
        ##          cost_expectations, expected_rankings, padding, costs)

        # recall that n is the number of agents, and N is the random demand we just generated
        # instead of a function call within our model, we will run the step to our agents:

        bids = np.zeros(self.n)

        for i, agent in enumerate(self.agents):
            bids[i] = agent.generate_bid(
                self.mechanism, N
            )  # note that all of our other params were initialized in the agent constructor

            # now we need to run the auction:
            ##     quantities, last_price = auction(N, bids, capacities)

            quantities, last_price = agent.run_auction(N, bids, self.capacities)

            # now we need to calculate the revenue for each agent:
            ##     revenues, profits = outcomes(mechanism, last_price, bids, quantities, costs)

            revenues, profits = agent.outcomes(
                self.mechanism, last_price, bids[i], quantities[i], self.costs[i]
            )

            # And then finally we need to clean up into our result arrays:
            ##     RTO_costs[s] = sum(revenues)
            ##     GenCo_profits[s] = profits

            self.RTO_costs[self.step_count] += revenues  # Accumulate revenues
            self.GenCo_profits[self.step_count, i] = profits  # Store individual profit

            agent.update_outcomes((revenues, profits))

        self.step_count += 1

    def run_model(self, mechanism: str = None) -> dict:
        """Run model for specified number of steps and return results

        Returns:
            dict: Results containing:
                'RTO_costs': Array of RTO costs for each step
                'GenCo_profits': 2D array of profits for each GenCo at each step
                'mechanism': Mechanism used for this run
        """

        # here we are going to do some initialization similar to what the source notebook does
        ##      costs = rng.uniform(min_cost, max_cost, n)
        ##      self.capacities = self.find_capacities(self.costs, self.m, self.b)
        ##      errors, paddings = self.create_expectations()

        self.costs = self.rng.uniform(self.min_cost, self.max_cost, self.n)
        self.capacities = self.find_capacities(
            self.costs, self.m, self.b
        )  # see function above

        ##  price_cap = max(costs) * cap_multiplier
        self.price_cap = max(self.costs) * self.cap_multiplier

        ##      errors = (rng.uniform(min_error, max_error, n*n) / 100 * 
        ##          np.random.choice([-1,1],size=n*n)).reshape([n,n])
        ##      padding = rng.uniform(min_padding, max_padding, n) / 100

        self.errors = (
            self.rng.uniform(self.min_error, self.max_error, self.n * self.n)
            / 100
            * np.random.choice([-1, 1], size=self.n * self.n)
        ).reshape([self.n, self.n])
        self.paddings = (
            self.rng.uniform(self.min_padding, self.max_padding, self.n) / 100
        )

        ##      capacity_expectations, cost_expectations, expected_rankings = 
        ##          create_expectations(n, capacities, costs)
        self.capacity_expectations, self.cost_expectations, self.expected_rankings = (
            self.create_expectations(self.n, self.capacities, self.costs)
        )  # see function above

        # before we run steps let's make sure we have a valid mechanism

        if mechanism is not None:
            if mechanism not in ["uniform", "discriminatory", "ownbid"]:
                raise ValueError(f"Invalid mechanism: {mechanism}")
            self.mechanism = mechanism

        # and now we can run the model

        self.step_count = 0
        for _ in range(self.steps):
            self.step()

        # Finally, we will return our "running tallies" of results into our expected 
        # dict of arrays.
        # a more Mesa-like approach would be simply to census the agents and return 
        # their "outcomes" accumulations.

        return {
            "RTO_costs": self.RTO_costs,
            "GenCo_profits": self.GenCo_profits,
            "mechanism": self.mechanism,
        }
