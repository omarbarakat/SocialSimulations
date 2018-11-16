# @author Omar Barakat
# @date 07-11-2018

# THIS is a simulation of the "tragedy of commons" (also would be very similar to a volunteer's dilemma with n individuals needed to volunteer in order for the good to remain functional).
# The main idea is that when a player cheats they lose the trust of [a portion of] the soceity.

# The addition to the game setting is the following:
# - The population starts with a common resource shared
# - At every turn an individual gets to normally use the resource (gets a small reward) or abuses it (gets a high reward)
# - if too many individuals abuse the resource it degrades, lowering the gains of all cases to all individuals
# - everytime an individual cheats a random portion of the soceity knows about it and have lower trust score in the individual
# - N individuals can decide to kill an individual, but if not enough individuals agree, the ones voted lose trust (as if they cheated in the resource)
# - If the individual dies they don't pass genes
# - decisions are based on:
#	1- aggregated trust score
#	2- state of the resource

# Environment settings:
# - The simulation is through a genetic algorithm. Generations of individuals are generated, get selected, pass their genes and evolve.
# - Every generation goes through a simulation of the environment with random pairs playing against eachother
# - The genes represent the probability to cheat at different "payoffs" of cheating. I.e. for values of E(payoff|cheat) - E(payoff|cooperate) genes will determine the tendency to cheat/cooperate
# - Fitness of the players is assessed in the end of the simulation based on the final wealth value
# - The individuals with the highest wealth value pass their genes, mate and mutate
# - For the final generation, the final graph for the probability given the payoff for an average individual will be plotted

# TODO:
# - try to expand to 2nd degree connections

from math import log, floor, ceil, sqrt
from enum import Enum
from random import randrange, random
from deap import creator, base, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

initial_wealth = 100.0
soceity_size = 50
num_connections = 10 					# NOT USED: maybe use it at some point to limit connections
num_distribution_bins = 100
num_generations = 500
num_games_per_simulation = 100*soceity_size	# average n games/individual
decisions = Enum('decisions', 'cheat cooperate')
gene_activity = Counter()

def get_distribution_bin_ind(value):
	bin_id = int(
		max(
			min( floor(value/30.0)+int(num_distribution_bins/2.0)
			, num_distribution_bins-1)
		, 0)
	)
	gene_activity[bin_id] += 1
	return bin_id


# wealth multipliers based on wealth of the players
# outcomes returned are in order: outcome[decisions.cheat][decisions.cooperate] means p1 cheated and p2 cooperated
# NOTE: these are absolute amounts: not symmetric between players
def get_game_payoffs(p1_wealth, p2_wealth):
	base_project_profit = 1.25
	participation_ratio = 0.2
	net_profit = participation_ratio*(p1_wealth+p2_wealth)*base_project_profit
	return {
		decisions.cheat: {
			decisions.cheat: -(0.5*participation_ratio)*p1_wealth,	# lose half the participated amount
			decisions.cooperate: net_profit							# get the whole profit!
		},
		decisions.cooperate: {
			decisions.cheat: -participation_ratio*p1_wealth,			# lose the participated amount
			decisions.cooperate: 0.5*net_profit							# split the profit
		},
	}


def get_knowledge_about(connections_ids, games, player_id, opponent_id):
	num_cooperate = 0.0
	num_cheat = 0.0
	for c in connections_ids+[player_id]:	# any game they or their connections played
		for g in games:
			if g['p1_id'] == c and g['p2_id'] == opponent_id:
				num_cooperate += (1 if g['p2_decision']==decisions.cooperate else 0)
				num_cheat += (1 if g['p2_decision']==decisions.cheat else 0)
			if g['p2_id'] == c and g['p1_id'] == opponent_id:
				num_cooperate += (1 if g['p1_decision']==decisions.cooperate else 0)
				num_cheat += (1 if g['p1_decision']==decisions.cheat else 0)
	return 0.5 if num_cooperate+num_cheat==0 else num_cooperate/(num_cooperate+num_cheat)


# use the probability distribution with the expected wealth return
# NOTE: could have optimized for expected fit change but assuming humans are greedy beings!
def make_decision(p1, trade_payoff, p1_trust_in_p2):
	expected_return_cooperating = 	(
			p1_trust_in_p2*trade_payoff[decisions.cooperate][decisions.cooperate]+
			(1-p1_trust_in_p2)*trade_payoff[decisions.cooperate][decisions.cheat]
		)
	expected_return_cheating = 		(
			p1_trust_in_p2*trade_payoff[decisions.cheat][decisions.cooperate]+
			(1-p1_trust_in_p2)*trade_payoff[decisions.cheat][decisions.cheat]
		)
	prob = p1[ get_distribution_bin_ind(expected_return_cheating - expected_return_cooperating) ]
	# Alternative:
	# 'goodness_dist_params': {'a':0, 'loc':0, 'scale':100},	# randomize parameters for beta distribution?
	# prob = skewnorm.pdf( expected_return_cheating - expected_return_cooperating, p1['goodness_dist_params']['a'], p1['goodness_dist_params']['loc'], p1['goodness_dist_params']['scale'])
	return decisions.cheat if prob > random() else decisions.cooperate 	# the decision is deterministic for now


# fitness function. Note: fitness is NOT linear in wealth.
def fitness(wealth):
	# return -1.0 if wealth<1 else log(wealth, 10)
	return wealth


def simulate_life(population, num_games):
	# games history (to represent each individual knowledge)
	# each should have the fields ['p1_id', 'p1_decision', 'p2_id', 'p2_decision', 'p1_wealth', 'p2_wealth']
	games_played = []
	# play n games
	for game_ind in range(num_games):
		# pick two players randomly
		p1_id = p2_id = randrange(len(population))
		while p2_id==p1_id:		p2_id = randrange(len(population))
		p1 = population[p1_id]
		p2 = population[p2_id]

		# PLAY THE GAME
		# calculate profits/losses
		trade_payoff_p1 = get_game_payoffs(p1.wealth, p2.wealth)
		trade_payoff_p2 = get_game_payoffs(p2.wealth, p1.wealth)
		# calculate trust
		p1_trust_in_p2 = get_knowledge_about(p1.connections_ids, games_played, p1_id, p2_id)
		p2_trust_in_p1 = get_knowledge_about(p1.connections_ids, games_played, p2_id, p1_id)
		# calculate decisions
		p1_decision = make_decision(p1, trade_payoff_p1, p1_trust_in_p2)
		p2_decision = make_decision(p2, trade_payoff_p2, p2_trust_in_p1)
		# save the game
		games_played.append( {
			'p1_id': p1_id, 'p1_decision': p1_decision, 'p1_wealth': p1.wealth, 
			'p2_id': p2_id, 'p2_decision': p2_decision, 'p2_wealth': p2.wealth
		} )
		# update wealth
		p1.wealth += trade_payoff_p1[p1_decision][p2_decision]
		p2.wealth += trade_payoff_p2[p2_decision][p1_decision]
		# update connections
		if p2_id in p1.connections_ids and p2_decision==decisions.cheat:		# remove p2 from p1 connections if cheated
			p1.connections_ids.remove(p2_id)
		if p1_id in p2.connections_ids and p1_decision==decisions.cheat:		# remove p1 from p2 connections if cheated
			p1.connections_ids.remove(p1_id)
		if p2_decision==decisions.cooperate and p1_decision==decisions.cooperate:	# if both cooperated: make friends
			if p2_id not in p1.connections_ids:	p1.connections_ids.append(p2_id)
			if p1_id not in p2.connections_ids:	p2.connections_ids.append(p1_id)
	return games_played


def init_env():
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list, wealth=initial_wealth, connections_ids=[], fitness=creator.FitnessMax)

	toolbox = base.Toolbox()
	toolbox.register("goodness_prob", random)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.goodness_prob, num_distribution_bins)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutFlipBit, indpb=0.005)
	toolbox.register("select", tools.selTournament, tournsize=3)
	return toolbox


def test_code():
	toolbox = init_env()
	population = toolbox.population(n=2)
	games = simulate_life(population, 1)
	for g in games:
		p1_payoffs = get_game_payoffs(g["p1_wealth"], g["p2_wealth"])
		p2_payoffs = get_game_payoffs(g["p2_wealth"], g["p1_wealth"])
		p1_outcome = p1_payoffs[g["p1_decision"]][g["p2_decision"]]
		p2_outcome = p2_payoffs[g["p2_decision"]][g["p1_decision"]]
		print(g)
		print("p1_outcome: "+str(p1_outcome))
		print("p2_outcome: "+str(p2_outcome))

	print("\n\n")
	for i, p in enumerate(population):
		print("player "+str(i))
		print("connections: "+str(p.connections_ids))
		print("wealth: "+str(p.wealth))
		print(p)


if __name__ == "__main__":
	print("[log] initializing")
	toolbox = init_env()

	print("[log] generating population")
	population = toolbox.population(n=soceity_size)

	for gen in range(num_generations):
		print("[log] simulating the world! processing generation: "+str(gen))
		offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
		games = simulate_life(offspring, num_games_per_simulation)
		for ind in offspring:
			ind.fitness.values = (fitness(ind.wealth),)
		selected = toolbox.select(offspring, k=len(population))
		population = [toolbox.clone(ind) for ind in selected]
	print("[log] Done simulating")

	topK = tools.selBest(population, k=5)						# get top K individuals from the final generation
	topKMean = np.sum(np.array(topK), axis=0)*1.0/len(topK)		# compute the average values for genes among top K
	print("top gene values: "); print(topK[0])
	print("effective genes: "); print([gene_activity[i] for i in range(num_distribution_bins)])
	
	plt.figure(1)
	for i, p in enumerate(topK):
		plt.subplot( ceil(sqrt(len(topK))), ceil(sqrt(len(topK))), i+1)
		plt.plot(p)
	plt.show()