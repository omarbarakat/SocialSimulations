# @author Omar Barakat
# @date 23-09-2018

# THIS is a simulation of the "social prisoner's dilemma". The main idea is that when a player cheats they not only lose the trust of the oponent, but also the trust of their "connections"

# The addition to the game setting is the following:
# - Players can share knowledge about other individuals through social connections (although you can enable global sharing of knowledge using the flag use_global_knowledge)
# - Player form social connection when both get involved in a game and both play fair
# - Player loses existing social connection when any of them cheat
# - Knowledge of a player about trustworthiness of another is based on games they or their connections played against the other
# - Each player has wealth value, the more wealthy an individual, the more profitable it is to trade with
# - Fitness is NOT linear in wealth. Using log to make it saturate slowly.

# Environment settings:
# - The simulation is through a genetic algorithm. Generations of individuals are generated, get selected, pass their genes and evolve.
# - Every generation goes through a simulation of the environment with random pairs playing against eachother
# - The genes represent the probability to cheat at different "payoffs" of cheating. I.e. for values of E(payoff|cheat) - E(payoff|cooperate) genes will determine the tendency to cheat/cooperate
# - Fitness of the players is assessed in the end of the simulation based on the final wealth value
# - The individuals with the highest wealth value pass their genes, mate and mutate
# - For the final generation, the final graph for the probability given the payoff for an average individual will be plotted

# TODO:
# - track behaviors of different segments of the soceity (P[cheat] over generations): Will be super interesting with inheritance!

from math import log, floor, ceil, sqrt
from enum import Enum
from random import randrange, random
from deap import creator, base, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import sys

# Parameters
initial_wealth = 100.0
soceity_size = 50
num_connections = 10 			# TODO [NOT USED] maybe use it at some point to limit connections
num_distribution_bins = 100
num_generations = 500
num_games_per_simulation = 100*soceity_size	# average n games/individual
use_synth_pop = False
use_global_knowledge = True
use_inheritance = False
if use_synth_pop: num_games_per_simulation = 3*soceity_size

# Data containers
decisions = Enum('decisions', 'cheat cooperate')
gene_activity = [0]*num_distribution_bins
num_segments = 3
segments_trace = defaultdict(lambda: {'wealth': [], 'goodness': []})
class_transition = {'higher': [], 'lower': []}
decision_distribution = {'both_cheat': [], 'one_cheats': [], 'both_coop': []}

def get_distribution_bin_ind(value):
	bin_id = int(
		max(
			min( floor(value/1.0)+int(num_distribution_bins/2.0)
			, num_distribution_bins-1)
		, 0)
	)
	gene_activity[bin_id] += 1
	return bin_id


# wealth multipliers based on wealth of the players
# outcomes returned are in order: outcome[decisions.cheat][decisions.cooperate] means p1 cheated and p2 cooperated
# The multiplers are the same as those of the prisoners dilemma +3. Refer to wikipedia. Notice, using these numbers make the expectation reduce to (P_cheat-P_coop)
def get_game_payoffs(p1_wealth, p2_wealth):
	scale = 40
	# # make scale dependent on the wealth
	# scale = 0.2*(p1_wealth+p2_wealth)
	return {
		decisions.cheat: {
			decisions.cheat: 0*scale,
			decisions.cooperate: 1*scale
		},
		decisions.cooperate: {
			decisions.cheat: -1*scale,
			decisions.cooperate: 0.5*scale
		},
	}

def get_knowledge_about(connections_ids, games, player_id, opponent_id):
	num_cooperate = 0.0
	num_cheat = 0.0
	for g in games:
		if not use_global_knowledge and g['p1_id'] not in connections_ids+[player_id] and g['p2_id'] not in connections_ids+[player_id]:
			continue
		if g['p2_id'] == opponent_id:
			num_cooperate += (1 if g['p2_decision']==decisions.cooperate else 0)
			num_cheat += (1 if g['p2_decision']==decisions.cheat else 0)
		if g['p1_id'] == opponent_id:
			num_cooperate += (1 if g['p1_decision']==decisions.cooperate else 0)
			num_cheat += (1 if g['p1_decision']==decisions.cheat else 0)
	return 0.5 if num_cooperate+num_cheat==0 else 1.0*num_cooperate/(num_cooperate+num_cheat)


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
	return decisions.cheat if prob > random() else decisions.cooperate 	# the decision is deterministic for now

# This function updates the average wealth and goodness of every segment of the soceity. Helpful to track if you get a bourgeois and proletariat segments at some point
parents_segments = [0]*soceity_size		# used for contrasting the progression of generations
def update_soceity_stats(population, games):
	global parents_segments
	new_segments = [0]*soceity_size
	cnt_stepped_up = 0
	cnt_stepped_down = 0
	sorted_pop = sorted(population, key=lambda i: i.wealth, reverse=True)
	seg_total_wealth = [0]*num_segments
	seg_total_goodness = [0]*num_segments
	i=0
	for i in range(len(sorted_pop)):	# loop on individuals
		seg = int(i/(len(sorted_pop)/num_segments))
		ind = sorted_pop[i]
		seg_total_wealth[seg] += ind.wealth
		seg_total_goodness[seg] += np.mean(ind)
		new_segments[ind.id] = seg
		cnt_stepped_up   += (1 if seg>min(parents_segments[ind.parents[0]], parents_segments[ind.parents[1]]) else 0)
		cnt_stepped_down += (1 if seg<max(parents_segments[ind.parents[0]], parents_segments[ind.parents[1]]) else 0)
	class_transition['higher'].append(cnt_stepped_up/soceity_size)
	class_transition['lower'].append(-cnt_stepped_down/soceity_size)
	parents_segments = new_segments
	for s in range(num_segments):		# loop on segmens and append stats
		segments_trace[s]['wealth'].append( seg_total_wealth[s]/len(sorted_pop) )
		segments_trace[s]['goodness'].append( seg_total_goodness[s]/len(sorted_pop) )
	one_cheats=0; both_cheat=0; both_coop=0;
	for g in games:
		if g['p1_decision']!=g['p2_decision']: one_cheats+=1.0/num_games_per_simulation
		elif g['p1_decision']==decisions.cheat: both_cheat+=1.0/num_games_per_simulation
		else: both_coop+=1.0/num_games_per_simulation
	decision_distribution['one_cheats'].append(one_cheats)
	decision_distribution['both_cheat'].append(both_cheat)
	decision_distribution['both_coop'].append(both_coop)
	
# fitness function
def fitness(wealth):
	return wealth


def simulate_life(population, num_games, foreign_population=None):
	# games history (to represent each individual knowledge)
	# each should have the fields ['p1_id', 'p1_decision', 'p2_id', 'p2_decision', 'p1_wealth', 'p2_wealth']
	games_played = []
	for i, ind in enumerate(population):	
		ind.wealth = ind.wealth if use_inheritance else initial_wealth
		ind.connections_ids = []
		ind.id = i

	# play n games
	for game_ind in range(num_games):
		# pick two players randomly
		p1_id = p2_id = randrange(len(population))
		p1 = population[p1_id]
		if foreign_population != None:
			p2_id = randrange(len(foreign_population))
			p2 = foreign_population[p2_id]
		else:
			while p2_id==p1_id:		p2_id = randrange(len(population))
			p2 = population[p2_id]

		# PLAY THE GAME
		# calculate profits/losses
		trade_payoff_p1 = get_game_payoffs(p1.wealth, p2.wealth)
		trade_payoff_p2 = get_game_payoffs(p2.wealth, p1.wealth)
		# calculate trust
		p1_trust_in_p2 = get_knowledge_about(p1.connections_ids, games_played, p1_id, p2_id)
		p2_trust_in_p1 = get_knowledge_about(p2.connections_ids, games_played, p2_id, p1_id)
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
		if p2_id in p1.connections_ids and (p2_decision==decisions.cheat or p1_decision==decisions.cheat):		# remove p2 from p1 connections if cheated
			p1.connections_ids.remove(p2_id)
		if p1_id in p2.connections_ids and (p1_decision==decisions.cheat or p2_decision==decisions.cheat):		# remove p1 from p2 connections if cheated
			p2.connections_ids.remove(p1_id)
		if p2_decision==decisions.cooperate and p1_decision==decisions.cooperate:	# if both cooperated: make friends
			if p2_id not in p1.connections_ids:	p1.connections_ids.append(p2_id)
			if p1_id not in p2.connections_ids:	p2.connections_ids.append(p1_id)
	return games_played

def simulate_synth_life(population, num_games):
	return simulate_life(population, num_games, foreign_population=[toolbox.good_individual(), toolbox.bad_individual()])


def cx(ind1, ind2):
	new_wealth = ( (ind1.wealth+ind2.wealth)/2 if use_inheritance else initial_wealth )
	parents = (ind1.id, ind2.id)
	tools.cxUniform(ind1, ind2, 0.5)   # tools.cxTwoPoint
	ind1.wealth = ind2.wealth = new_wealth
	ind1.connections_ids = []; ind2.connections_ids = []
	ind1.parents = ind2.parents = parents
	return ind1, ind2

def init_env():
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list, wealth=initial_wealth, connections_ids=[], parents=(0, 0), id=0, fitness=creator.FitnessMax)
	
	toolbox = base.Toolbox()
	toolbox.register("goodness_prob", random)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.goodness_prob, num_distribution_bins)
	toolbox.register("good_individual", tools.initRepeat, creator.Individual, lambda: 1, num_distribution_bins)
	toolbox.register("bad_individual", tools.initRepeat, creator.Individual, lambda: 0, num_distribution_bins)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	# toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mate", cx)
	toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
	toolbox.register("select", tools.selTournament, tournsize=3)
	return toolbox

def test_code():
	toolbox = init_env()
	population = toolbox.population(n=2)
	games = simulate_life(population, 10)
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
	print(gene_activity)


if __name__ == "__main__":
	print("====================\nSIMULATING THE WORLD\n====================\nsoceity_size=%d\nnum_connections=%d\nnum_generations=%d\nuse_synth_pop=%d\nuse_global_knowledge=%d\nuse_inheritance=%d\n====================\n" % \
		(soceity_size, num_connections, num_generations, use_synth_pop, use_global_knowledge, use_inheritance))
	
	print("[log] initializing")
	toolbox = init_env()

	# SIMULATE
	print("[log] generating population")
	population = toolbox.population(n=soceity_size)
	old_record = 0; repeated=0
	for gen in range(num_generations):
		print("[log] simulating the world! processing generation: "+str(gen))
		offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.05)
		
		games = simulate_synth_life(offspring, num_games_per_simulation) if use_synth_pop\
				else simulate_life(offspring, num_games_per_simulation)
		
		new_record = -sys.maxsize-1
		for ind in offspring:
			f = fitness(ind.wealth)
			ind.fitness.values = (f,)
			new_record = max(new_record, f)
		print("improvement = "+str(new_record-old_record)+" with gain "+str(new_record))
		repeated = repeated+1 if new_record==old_record else 0
		if repeated==3:
			print("CONVERGED")
			break
		old_record = new_record
		population = toolbox.select(offspring, k=len(population))
		update_soceity_stats(population, games)
	print("[log] Done simulating")

	# PRINT AND PLOT
	# Print genes of top individuals
	topK = tools.selBest(population, k=10)						# get top K individuals from the final generation
	topKMean = np.mean(np.array(topK), axis=0)					# compute the average values for genes among top K
	print("top gene values: "); print(topK[0])
	print("effective genes: "); print([gene_activity[i] for i in range(num_distribution_bins)])
	print("overall average goodness: "+str(np.mean(topKMean)))

	# Plot genes of top individuals
	first_active = next((i for i, x in enumerate(gene_activity) if x!=0), None)
	last_active  = len(gene_activity) - next((i for i, x in enumerate(gene_activity[::-1]) if x!=0), None)
	plt.figure("Top Individuals' Genes")
	for i, p in enumerate(topK):
		plt.subplot( ceil(sqrt(len(topK))), ceil(sqrt(len(topK))), i+1)
		plt.plot([c for c in range(first_active, last_active)], p[first_active:last_active])
	plt.show()

	# Plot progression of soceity segments
	# Plot wealth
	plt.figure("Progression over generations")
	plt.subplot(4, 1, 1)
	for i in range(num_segments):
		plt.plot(segments_trace[i]['wealth'], label='segment '+str(i))
	plt.plot([np.mean([segments_trace[j]['wealth'][i] for j in range(num_segments)]) for i in range(len(segments_trace[0]['wealth']))], label='overall wealth', linestyle=':')
	plt.title('wealth of segments over generations'); plt.legend();
	# Plot goodness
	plt.subplot(4, 1, 2)
	for i in range(num_segments):
		plt.plot(segments_trace[i]['goodness'], label='segment '+str(i))
	plt.plot([np.mean([segments_trace[j]['goodness'][i] for j in range(num_segments)]) for i in range(len(segments_trace[0]['goodness']))], label='overall goodness', linestyle=':')
	plt.title('goodness of segments over generations'); plt.legend(); 
	# Plot class transition
	plt.subplot(4, 1, 3)
	plt.plot(class_transition['higher'], 'g^', label='up')
	plt.plot(class_transition['lower'], 'rv', label='down')
	plt.title('class transition over generations'); plt.legend(); 
	# Plot decision distribution
	plt.subplot(4, 1, 4)
	plt.plot(decision_distribution['both_coop'], 'g', label='both_coop')
	plt.plot(decision_distribution['both_cheat'], 'r', label='both_cheat')
	plt.plot(decision_distribution['one_cheats'], 'y', label='one_cheats')
	plt.title('action distribution'); plt.legend(); 
	plt.show()