from mesa import Agent
import numpy as np

class GenCoAgent(Agent):
    """Generator Company agent that participates in electricity auctions"""
    
    def __init__(
        self, 
        unique_id: int, 
        model,
        cost: float,
        capacity: float,
        error_factors: np.ndarray,
        padding: float
    ):
        super().__init__(unique_id, model)
        self.cost = cost
        self.capacity = capacity
        self.error_factors = error_factors  # Expectations of other GenCos' costs
        self.padding = padding  # Bidding strategy parameter
        self.revenue = 0
        self.profit = 0
        self.quantity_supplied = 0
        self.outcomes = None
        
    def generate_bid(self, mechanism, N, capacity_expectations, cost_expectations, expected_rankings, padding, costs):
        """Generate bid based on mechanism type and demand"""
        # this function is the ALMOST the same as the one in the source notebook
        # we have slightly changed the function signature to match the agent's 
        # initialization (and capture the model-computed variables)
        # however, we are running on one agent at a time.
        # and we will just pass back a single bid.
        
        bid = None

        buffer = 0
        winners = []
        first_loser = None
        for r in expected_rankings:
            if buffer < N:
                buffer += capacity_expectations[r]
                winners.append(r)
            else:
                first_loser = r
                break
                
        if not winners:
            bid = self.model.price_cap
            return(bid)
            
        last_winner = winners[-1]
        
        # Generate bid based on mechanism
        if mechanism == 'uniform':
            if not first_loser:
                bid = self.model.price_cap
            elif self.unique_id == last_winner:
                bid = max(self.cost, 
                        cost_expectations[first_loser] * padding)
            else:
                bid = self.cost
                
        elif mechanism == 'discriminatory':
            if not first_loser:
                bid = self.model.price_cap # fix
            elif self.unique_id in winners:
                bid = max(self.cost, 
                        cost_expectations[first_loser] * padding)
            else:
                bid = self.cost
                
        else:  # ownbid
            bid = self.cost
            
        return(bid)
        

    def auction(self, N, bids, capacities):
        """Run auction and return quantities and last price"""
        # this function is the same as the one in the source notebook
        request = N
        quantities = np.zeros(n)
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

    # NOT from the source notebook, this plugs the outcome into the "body" of the agent.
    def update(self, outcome: tuple):
        """Update agent's revenue and profit"""
        self.outcomes.append(outcome)
