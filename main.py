# !/usr/bin/env python3
# main.py
# This serves as an entry point for running the model.
# This is an alternative to running the notebook.

import argparse

import numpy as np

from model import ElectricityMarket


def main():
    parser = argparse.ArgumentParser(description='Run Electricity Market Model')
    
    # Add arguments matching source notebook parameters
    parser.add_argument('--n', type=int, default=5, 
                        help='Number of generating companies')
    parser.add_argument('--m', type=float, default=-5, 
                        help='Slope of price-capacity relationship')
    parser.add_argument('--b', type=float, default=1000, 
                        help='Intercept of price-capacity relationship')
    parser.add_argument('--steps', type=int, default=1000, 
                        help='Number of steps to run')
    parser.add_argument('--mechanism', choices=['uniform', 'discriminatory', 'ownbid'], 
                        default='uniform', 
                        help='Pricing mechanism')
    
    args = parser.parse_args()
    
    # Create model with parameters
    params = vars(args)
    mechanism = params.pop('mechanism')  # Remove mechanism from constructor params
    
    model = ElectricityMarket(**params)
    results = model.run_model(mechanism)
    
    # Print summary statistics
    print(f"\nResults for {mechanism} mechanism:")
    print(f"Mean RTO cost: {np.mean(results['RTO_costs']):,.2f}")
    print(f"Mean GenCo profit: "
          f"{np.mean(np.sum(results['GenCo_profits'], axis=1)):,.2f}")

    # Print the total number of winning outcomes and average revenue and profit in winning outcomes for each agent
    print("\nWinning outcomes for each agent:")
    for i, agent_outcomes in enumerate(results['agent_outcomes']):
        winning_outcomes = [outcome for outcome in agent_outcomes if outcome['won_auction']]
        print(f"Agent {i} total wins: {len(winning_outcomes)}")
        print(f"Agent {i} average revenue: {np.mean([outcome['revenue'] for outcome in winning_outcomes]):,.2f}")
        print(f"Agent {i} average profit: {np.mean([outcome['profit'] for outcome in winning_outcomes]):,.2f}")
                

if __name__ == '__main__':
    main()