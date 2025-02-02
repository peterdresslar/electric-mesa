import numpy as np
from mesa import Agent


class GenCoAgent(Agent):
    """Generator Company agent that participates in electricity auctions"""
    
    def __init__(
        self, 
        model,
        mechanism: str,
        i_n: int
    ):
        super().__init__(model)
        self.revenue = 0
        self.profit = 0
        self.quantity_supplied = 0
        self.outcomes = []
        self.i_n = i_n # the ith agent of n agents

    # The agent-specific chunk of generate_bids() related to the bid
    def generate_bid(
        self, 
        mechanism, 
        last_winner,
        winners,
        first_loser,
        price_cap,
        costs,
        cost_expectations,
        paddings,
    ):
        """Generate a bid for the agent based on market position and expectations
        
        Args:
            mechanism: Pricing mechanism ('uniform', 'discriminatory', or 'ownbid')
            last_winner: Index of last winning agent in expected rankings
            winners: List of winning agent indices
            first_loser: Index of first losing agent
            price_cap: Maximum allowed bid price
            costs: Array of all agent costs
            cost_expectations: 2D array of cost expectations (agent i's expectations for agent j)
            paddings: Array of padding factors for each agent
        """
        i_n = self.i_n  # for convenience

        # the following code is the same as the source notebook
        if mechanism == 'uniform':
            if not first_loser:
                bid = price_cap
            elif i_n == last_winner:
                bid = max(
                    costs[i_n], cost_expectations[i_n, first_loser] * paddings[i_n])
            else:
                bid = costs[i_n]
                
        elif mechanism == 'discriminatory':
            if not first_loser:
                bid = price_cap
            elif i_n in winners:
                bid = max(
                    costs[i_n], cost_expectations[i_n, first_loser] * paddings[i_n])
            else:
                bid = costs[i_n]
                
        else:  # mechanism == 'ownbid'
            bid = costs[i_n]
            
        return bid

    def update_outcomes(self, revenues, profits):
        """Update agent's revenue and profit for this step"""
        self.revenue = revenues[self.i_n]
        self.profit = profits[self.i_n]
        self.outcomes.append({
            'step': self.model.step_count,
            'revenue': self.revenue,
            'profit': self.profit,
            'won_auction': self.revenue > 0  # Track if agent won this round
        })