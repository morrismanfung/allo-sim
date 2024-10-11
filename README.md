# Asset Allocation Simulation

Author: Morris Chan

This project aims to build a simple tool kit for simulating performance of asset allocation strategies. One popular allocation strategy consist of 60% equity and 40% bond, rebalanced quarterly. Some says the proportion of bond should be equal to one's age or 20 minus one's age, i.e., a 24 year-old should hold 24% or 4% bond in their investment portfolio. This project will provide basic functions for simulating return and risk with the specified assets.

## Model

Gaussian Hidden Markov Model and Gaussian Mixture Hidden Markov Model are used in this project. It is assumed that asset's performance is dependant on an unobserved state (similar to the concept of bull vs bear market). A Markov model will be trained using historical data and used to simulate future performance by random sampling.