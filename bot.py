import random
from collections import Counter, deque
import copy
import numpy as np
from itertools import combinations
import nash
import networkx as nx
import ast
from game import *

class State:
    def __init__(self, h, k, s, p):
        self.hands = h
        self.kept_cards = k
        self.scores = s
        self.puddings = p
    def __str__(self):
        return "Hands: {}\nKept Cards: {}\nScores: {}\nPuddings: {}".format(self.hands, self.kept_cards, self.scores, self.puddings)
    def __repr__(self):
        return "Hands: {}\nKept Cards: {}\nScores: {}\nPuddings: {}".format(self.hands, self.kept_cards, self.scores, self.puddings)

def string_to_state(string):
    spl = string.split("\n")
    h = deque(ast.literal_eval(spl[0][13:-1]))
    k = ast.literal_eval(spl[1][12:])
    s = ast.literal_eval(spl[2][8:])
    p = ast.literal_eval(spl[3][10:])
    return State(h, k, s, p)

def convert_to_choice_list(string):
    val = string.split(",")
    if type(val) != list:
        val = [val]
    if len(val) == 2:
        val[1].lstrip()
    return val

# ugly, I should tidy this function later
def possible_choices(state, self_index):
    hands = state.hands
    kept_cards = state.kept_cards
    scores = state.scores
    puddings = state.puddings

    my_hand = hands[self_index]
    if 'Chopsticks' not in kept_cards[self_index] or (len(my_hand) < 2):
        return list(map(convert_to_choice_list, list(set(my_hand))))
    else:
        counts = [i for i in Counter(my_hand).items()]
        two_same_card_choices = []
        for count in counts:
            if count[1] >= 2:
                two_same_card_choices.append(count[0] + ',' + count[0])
        two_distinct_card_choices = [",".join(map(str, comb)) for comb in combinations(set(my_hand), 2)]
        two_distinct_card_choices = list(map(convert_to_choice_list, two_distinct_card_choices))
        two_same_card_choices = list(map(convert_to_choice_list, two_same_card_choices))
        one_card_choices = list(set(my_hand))
        one_card_choices = list(map(convert_to_choice_list, one_card_choices))
        return one_card_choices + two_same_card_choices + two_distinct_card_choices

# should be synced with similar code in play_round
def choice_result(card_choices_in, state):
    card_choices = copy.deepcopy(card_choices_in)
    hands = copy.deepcopy(state.hands)
    kept_cards = copy.deepcopy(state.kept_cards)
    scores = copy.deepcopy(state.scores)
    puddings = copy.deepcopy(state.puddings)
    n_players = len(hands)
    for i in range(n_players):
        choice = card_choices[i]
        for j in range(len(choice)):
            kept_cards[i].append(choice[j])
            hands[i].remove(choice[j])
        if len(choice) == 2:
            assert 'Chopsticks' in kept_cards[i]
            hands[i].append('Chopsticks')
            kept_cards[i].remove('Chopsticks')
        hands[i].sort()
    hands.rotate(1)
    return State(hands, kept_cards, scores, puddings)

# built specifically for 2 players
def possible_results(state):
    n_players = len(state.hands)
    poss_choices = [possible_choices(state, i) for i in range(n_players)]
    # n_poss_results = np.prod(np.array(list(map(len, poss_moves))))
    n_1 = len(poss_choices[0])
    n_2 = len(poss_choices[1])
    poss_results = [None] * (n_1 * n_2)
    for i in range(n_1):
        for j in range(n_2):
            index = i * n_2 + j
            poss_results[index] = choice_result([poss_choices[0][i], poss_choices[1][j]], state)
    return poss_results, poss_choices

def make_graph3(state):
    g = nx.DiGraph()
    g.add_node(state.__str__())
    successors_1, _ = possible_results(state)
    str_successors_1 = list(map(str, successors_1))
    g.add_nodes_from(str_successors_1)
    g.add_star([str(state)] + str_successors_1)
    for succ_1 in successors_1:
        successors_2, _ = possible_results(succ_1)
        str_successors_2 = list(map(str, successors_2))
        g.add_nodes_from(str_successors_2)
        g.add_star([str(succ_1)] + str_successors_2)
        for succ_2 in successors_2:
            successors_3, _ = possible_results(succ_2)
            str_successors_3 = list(map(str, successors_3))
            g.add_nodes_from(str_successors_3)
            g.add_star([str(succ_2)] + str_successors_3)
    return g

def make_graph4(state):
    g = nx.DiGraph()
    g.add_node(state.__str__())
    successors_1, _ = possible_results(state)
    str_successors_1 = list(map(str, successors_1))
    g.add_nodes_from(str_successors_1)
    g.add_star([str(state)] + str_successors_1)
    for succ_1 in successors_1:
        successors_2, _ = possible_results(succ_1)
        str_successors_2 = list(map(str, successors_2))
        g.add_nodes_from(str_successors_2)
        g.add_star([str(succ_1)] + str_successors_2)
        for succ_2 in successors_2:
            successors_3, _ = possible_results(succ_2)
            str_successors_3 = list(map(str, successors_3))
            g.add_nodes_from(str_successors_3)
            g.add_star([str(succ_2)] + str_successors_3)
            for succ_3 in successors_3:
                successors_4, _ = possible_results(succ_3)
                str_successors_4 = list(map(str, successors_4))
                g.add_nodes_from(str_successors_4)
                g.add_star([str(succ_3)] + str_successors_4)
    return g

unique_cards = sorted(list(set(default_cards_list)))
# first the unique_cards in player 1's hand, (player 2's values are the reverse),
# then unique_cards in player 1's hand squared, (player 2's values are the reverse),
# (except with sashimi and tempura being calculated mod 3 and mod 2, respectively)
# then a feature calculating the point differential if the game ended now
default_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0,
                   1]
# positive for player 1, negative for player 2
def linear_eval(state, weights):
    if type(state) == str:
        state = string_to_state(state)
    assert isinstance(state, State)
    p1_counter = Counter(state.kept_cards[0])
    p1_counter['Sashimi'] = p1_counter['Sashimi'] % 3
    p1_counter['Tempura'] = p1_counter['Tempura'] % 2
    p2_counter = Counter(state.kept_cards[1])
    p2_counter['Sashimi'] = p2_counter['Sashimi'] % 3
    p2_counter['Tempura'] = p2_counter['Tempura'] % 2
    p1_counts = [p1_counter[card] for card in unique_cards]
    p2_counts = [p2_counter[card] for card in unique_cards]
    p1_counts_sq = list(np.array(p1_counts) * np.array(p1_counts))
    p2_counts_sq = list(np.array(p2_counts) * np.array(p2_counts))
    scores = calculate_score(state.kept_cards)
    features = np.array(p1_counts + p1_counts_sq + [scores[0]]) - np.array(p2_counts + p2_counts_sq + [scores[1]])
    return np.sum(features * np.array(weights))

# only use this function WITHIN a nash tree as created in nash_tree_X
def get_nash_game(graph, state, poss_choices, weights):
    choices1 = poss_choices[0]
    choices2 = poss_choices[1]
    arr = np.zeros((len(choices1), len(choices2)))
    for i in range(len(choices1)):
        c1 = choices1[i]
        for j in range(len(choices2)):
            c2 = choices2[j]
            result_state = choice_result([c1, c2], state)
            arr[i][j] = graph.node[str(result_state)]['value']
    return nash.Game(arr)

def get_nash_value(graph, state, poss_choices, weights):
    game = get_nash_game(graph, state, poss_choices, weights)
    eqs = game.support_enumeration().__next__() # TODO: just uses the first nash equilibrium. Should I change this?
    return game[eqs[0], eqs[1]][0]

def nash_tree_3(state, weights):
    g = nx.DiGraph()
    g.add_node(state.__str__())
    successors_1, poss_choices_0 = possible_results(state)
    str_successors_1 = list(map(str, successors_1))
    g.add_nodes_from(str_successors_1)
    g.add_star([str(state)] + str_successors_1)
    for succ_1 in successors_1:
        successors_2, poss_choices_1 = possible_results(succ_1)
        str_successors_2 = list(map(str, successors_2))
        g.add_nodes_from(str_successors_2)
        g.add_star([str(succ_1)] + str_successors_2)
        for succ_2 in successors_2:
            successors_3, poss_choices_2 = possible_results(succ_2)
            str_successors_3 = list(map(str, successors_3))
            for str_succ_3 in str_successors_3:
                g.add_node(str_succ_3, value = linear_eval(str_succ_3, weights))
            g.add_star([str(succ_2)] + str_successors_3)
            g.node[str(succ_2)]['value'] = get_nash_value(g, succ_2, poss_choices_2, weights)
        g.node[str(succ_1)]['value'] = get_nash_value(g, succ_1, poss_choices_1, weights)
    root_game = get_nash_game(g, state, poss_choices_0, weights)
    eqs = root_game.support_enumeration().__next__()
    g.node[str(state)]['value'] = root_game[eqs[0], eqs[1]][0]
    # g.node[str(state)]['value'] = get_nash_value(g, state, poss_choices_0, weights)
    g.graph['policy'] = eqs
    return g

def nash_tree_4(state, weights):
    g = nx.DiGraph()
    g.add_node(state.__str__())
    successors_1, poss_choices_0 = possible_results(state)
    str_successors_1 = list(map(str, successors_1))
    g.add_nodes_from(str_successors_1)
    g.add_star([str(state)] + str_successors_1)
    for succ_1 in successors_1:
        successors_2, poss_choices_1 = possible_results(succ_1)
        str_successors_2 = list(map(str, successors_2))
        g.add_nodes_from(str_successors_2)
        g.add_star([str(succ_1)] + str_successors_2)
        for succ_2 in successors_2:
            successors_3, poss_choices_2 = possible_results(succ_2)
            str_successors_3 = list(map(str, successors_3))
            g.add_nodes_from(str_successors_3)
            g.add_star([str(succ_2)] + str_successors_3)
            for succ_3 in successors_3:
                successors_4, poss_choices_3 = possible_results(succ_3)
                str_successors_4 = list(map(str, successors_4))
                for str_succ_4 in str_successors_4:
                    g.add_node(str_succ_4, value = linear_eval(str_succ_4, weights))
                g.add_star([str(succ_3)] + str_successors_4)
                g.node[str(succ_3)]['value'] = get_nash_value(g, succ_3, poss_choices_3, weights)
            g.node[str(succ_2)]['value'] = get_nash_value(g, succ_2, poss_choices_2, weights)
        g.node[str(succ_1)]['value'] = get_nash_value(g, succ_1, poss_choices_1, weights)
    root_game = get_nash_game(g, state, poss_choices_0, weights)
    eqs = root_game.support_enumeration().__next__()
    g.node[str(state)]['value'] = root_game[eqs[0], eqs[1]][0]
    # g.node[str(state)]['value'] = get_nash_value(g, state, poss_choices_0, weights)
    g.graph['policy'] = eqs
    return g

def get_nash_game_recur(graph, state, poss_choices = None, weights = default_weights):
    n_players = 2
    if poss_choices is None:
        poss_choices = [possible_choices(state, i) for i in range(n_players)]
    choices1 = poss_choices[0]
    choices2 = poss_choices[1]
    arr = np.zeros((len(choices1), len(choices2)))
    for i in range(len(choices1)):
        c1 = choices1[i]
        for j in range(len(choices2)):
            c2 = choices2[j]
            result_state = choice_result([c1, c2], state)
            _, poss_c = possible_results(result_state)
            arr[i][j] = get_nash_value_recur(graph, result_state, poss_c, weights)
    return nash.Game(arr)

def get_nash_value_recur(graph, state, poss_choices = None, weights = default_weights):
    if 'value' in graph.node[str(state)]:
        return graph.node[str(state)]['value']
    if len(state.hands[0]) == 0:
        return linear_eval(state, weights)
    else:
        game = get_nash_game_recur(graph, state, poss_choices, weights)
        eqs = game.support_enumeration().__next__()  # TODO: just uses the first nash equilibrium. Should I change this?
        return game[eqs[0], eqs[1]][0]

got_to_5_cards = False
got_to_7_cards = False
def make_nash_tree_recur(graph, state, weights):
    global got_to_5_cards
    successors, possible_choices = possible_results(state)
    if len(state.hands[0]) == 5 and got_to_5_cards == False:
        print("got to 5 cards!!!!")
        got_to_5_cards = True
    if len(successors) == 0:
        graph.add_node(str(state), value = linear_eval(state, weights))
    else:
        graph.add_node(str(state))
        str_successors = list(map(str, successors))
        graph.add_nodes_from(str_successors)
        graph.add_star([str(state)] + successors)
        for succ in successors:
            make_nash_tree_recur(graph, succ, weights)
        graph.node[str(state)]['value'] = get_nash_value_recur(graph, state, possible_choices, weights)

def nash_tree_complete(state, weights):
    g = nx.DiGraph()
    g.add_node(str(state))
    make_nash_tree_recur(g, state, weights)
    game = get_nash_game_recur(g, state, None, weights)
    eqs = game.support_enumeration().__next__()
    g.graph['policy'] = eqs
    return g

h = deque([['Tempura', 'Tempura', 'Wasabi', 'Wasabi'], ['Chopsticks', 'Maki 1', 'Maki 2', 'Sashimi']])
k = [['Chopsticks'], ['Wasabi']]
k2 = [['Tempura', 'Tempura'], ['Wasabi', 'Salmon Nigiri']]
s = [0, 0]
p = [0, 0]
state1 = State(h, k, s, p)
possible_results(state1)

h1 = ['Egg Nigiri', 'Maki 2', 'Maki 3', 'Pudding', 'Pudding', 'Salmon Nigiri', 'Salmon Nigiri', 'Sashimi', 'Sashimi', 'Wasabi']
h2 = ['Chopsticks', 'Egg Nigiri', 'Maki 1', 'Maki 2', 'Maki 3', 'Pudding', 'Sashimi', 'Sashimi', 'Squid Nigiri', 'Tempura']

state_full = State(deque([h1, h2]), [[], []], s, p)

import time
begin = time.time()
n = nash_tree_complete(state_full, default_weights)
end = time.time()
print('time elapsed: ', end - begin)