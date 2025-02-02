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

        # Initialize arrays that will be populated in run_model()
        self.costs = None
        self.capacities = None
        self.price_cap = None
        self.errors = None
        self.paddings = None # we depart from the source notebook with a pluralization.

        # Add results tracking
        self.RTO_costs = np.zeros(self.steps)  # Store RTO costs for each step
        self.GenCo_profits = np.zeros((self.steps, self.n))  # Store profits for each GenCo

        # Initialize step counter
        self.step_count = 0

    ### Helper functions ###
    # The following two functions are taken from the source notebook
    # and are used here to set up vectors that will be passed into the agent
    # via step functions.
    ##############################################################

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
    
    def auction(self, N, bids, capacities):
        """Run auction and return quantities and last price"""
        # this function is the same as the one in the source notebook
        request = N
        quantities = np.zeros(self.n) # slightly changed to get the number of agents
        bid_ranking = np.argsort(bids)
        for i in bid_ranking:
            if request > 0:
                q = capacities[i]
                if q < request:
                    request -= q
                    quantities[i] = q
                else: 
                    quantities[i] = request
                    request = 0 
                    last_price = bids[i]
        return(quantities, last_price)


    def outcomes(self, mechanism, last_price, bids, quantities, costs):
        """Calculate revenue and profit for the agent"""
        # this function is the same as the one in the source notebook
        if mechanism == 'uniform':
            revenues = last_price * quantities  
        else:
            revenues = bids * quantities
        profits = revenues - costs * quantities
        return(revenues, profits)
    
    ### The Step Function ###
    # Here we will run one step of the model.
    ##############################################################

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

        # Here, we will need to deart from the source notebook substatially,
        # We have n agents with individual "generate_bid" logic.
        # The following code will organize the market side of bid generation.

        bids = np.zeros(self.n)
        for i_n, agent in enumerate(self.agents):
            buffer = 0
            winners = []
            first_loser = None
            for r in self.expected_rankings[i_n]:
                if buffer < N:
                    buffer += self.capacities[r]
                    winners.append(r)
                else:
                    first_loser = r
                    break
            last_winner = winners[-1]

            # now we use that market information to send to the agent to process a bid.

            bids[i_n] = agent.generate_bid(
                self.mechanism,
                last_winner,
                winners,
                first_loser,
                self.price_cap,
                self.costs,
                self.cost_expectations,
                self.paddings,
            )

            # now we need to run the auction:
            ##     quantities, last_price = auction(N, bids, capacities)

        quantities, last_price = self.auction(N, bids, self.capacities)

            # now we need to calculate the revenue for each agent:
            ##     revenues, profits = outcomes(mechanism, last_price, bids, quantities, costs)

        revenues, profits = self.outcomes(
            mechanism=self.mechanism,
            last_price=last_price,
            bids=bids,
            quantities=quantities,
            costs=self.costs
        )  # This line is pretty tricky, since we have a different cardinality of params

        # And then finally we need to clean up into our result arrays:
        ##     RTO_costs[s] = sum(revenues)
        ##     GenCo_profits[s] = profits

        self.RTO_costs[self.step_count] = np.sum(revenues)  # Accumulate revenues
        self.GenCo_profits[self.step_count] = profits  # Store individual profit

        # let agents record their individual results
        for i_n, agent in enumerate(self.agents):
            agent.update_outcomes(revenues[i_n], profits[i_n])

        self.step_count += 1

    def run_model(self, mechanism: str = None) -> dict:
        """Run model for specified number of steps and return results

        Returns:
            dict: Results containing:
                'RTO_costs': Array of RTO costs for each step
                'GenCo_profits': 2D array of profits for each GenCo at each step
                'mechanism': Mechanism used for this run
        """

        # To start the model run,
        # we are going to do some initialization similar to the source notebook
        ##      costs = rng.uniform(min_cost, max_cost, n)
        ##      self.capacities = self.find_capacities(self.costs, self.m, self.b)
        ##      errors, paddings = self.create_expectations()
        # however, rather than storing the results into arrays belonging to the model.
        # we will "endow" our agents as we go along

        # create the vector (ndarray) of costs. we store all of these vectors in the model
        # so that we can pass them into the agent via step functions.
        self.costs = self.rng.uniform(self.min_cost, self.max_cost, self.n)

        # create the vector (ndarray) of capacities
        self.capacities = self.find_capacities(
            self.costs, self.m, self.b
        )  # see function above

        ##  price_cap = max(costs) * cap_multiplier
        self.price_cap = max(self.costs) * self.cap_multiplier

        ##      errors = (rng.uniform(min_error, max_error, n*n) / 100 
        ##          * np.random.choice([-1,1],size=n*n)).reshape([n,n])
        ##      padding = rng.uniform(min_padding, max_padding, n) / 100

        self.errors = (
            self.rng.uniform(self.min_error, self.max_error, self.n * self.n) / 100
                * np.random.choice([-1, 1], size=self.n * self.n)).reshape([self.n, self.n])    # noqa
        self.paddings = self.rng.uniform(self.min_padding, self.max_padding, self.n) / 100      # noqa


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

        # Great. Now we can create our agents:
        # Adding in any settings that we have not already set into vectorized arrays

        for i_n in range(self.n):
            agent = GenCoAgent(
                self,
                self.mechanism,
                i_n=i_n
            )
            self.register_agent(agent)

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
