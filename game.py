import random
from collections import Counter, deque
import copy
import numpy as np
from itertools import combinations
import nash
import networkx as nx

class Deck:

    def __init__(self, cards):
        self.cards = cards
        random.shuffle(self.cards)

    def __repr__(self):
        return "Deck with cards: " + repr(self.cards)

    def draw(self, n):
        assert len(self.cards) >= n
        b = self.cards[:n].copy()
        del self.cards[:9]
        return b

def deck_list_from_dict(input_dict):
    output = []
    for item in input_dict.items():
        card = item[0]
        n_copies = item[1]
        output.extend([card] * n_copies)
    return output

default_cards_dict = {'Tempura': 14, 'Sashimi': 14, 'Dumpling': 14, 'Maki 2': 12, 'Maki 3': 8,
                      'Maki 1': 6, 'Salmon Nigiri': 10, 'Squid Nigiri': 5, 'Egg Nigiri': 5,
                      'Pudding': 10, 'Wasabi': 6, 'Chopsticks': 4}
default_cards_list = deck_list_from_dict(default_cards_dict)

# picks randomly, never uses chopsticks
def default_decision_function(hands, kept_cards, scores, puddings, self_index):
    return random.sample(hands[self_index], 1)

def human_decision_function(hands, kept_cards, scores, puddings, self_index):
    print("\nHi! You are player", self_index + 1)
    print("Currently your hand is ", hands[self_index])
    print("And the kept_cards are: ", kept_cards)
    while True:
        choice = input("Please input your choice: ")
        val = choice.split(",")
        if len(val) == 2:
            val[1] = val[1].lstrip()
        if len(val) == 1:
            if val[0] in hands[self_index]:
                print("Your choice was: ", val)
                return val
            else:
                print("Invalid choice: that card is not in your hand. Please pick again...")
        else:
            if 'Chopsticks' not in kept_cards[self_index]:
                print("Invalid choice, please pick again. You can only pick two cards if you have Chopsticks")
            if val[0] in hands[self_index]:
                new_hand = copy.deepcopy(hands[self_index])
                new_hand.remove(val[0])
                if val[1] in new_hand:
                    print("Using your Chopsticks, your choice was two cards: ", val)
                    return val
            print("Invalid choice. Please pick again...")

def second_largest(numbers):
    first, second = 0, 0
    for n in numbers:
        if n > first:
            first, second = n, first
        elif first > n > second:
            second = n
    return second

# TODO inefficient: optimize later!
def calculate_maki(maki_counts):
    n = len(maki_counts)
    maki_max = max(maki_counts)
    maki_second = second_largest(maki_counts)
    maki_max_count = maki_counts.count(maki_max)
    maki_second_count = maki_counts.count(maki_second)
    maki_scores = [0] * n

    for i in range(n):
        if maki_counts[i] == maki_max:
            maki_scores[i] = 6 // maki_max_count
        elif maki_counts[i] == maki_second:
            maki_scores[i] = 3 // maki_second_count
        else:
            maki_scores[i] = 0
    return maki_scores

def calculate_nigiri(card_sequence):
    wasabi_count = 0
    score = 0
    for card in card_sequence:
        if card == "Wasabi":
            wasabi_count = wasabi_count + 1
        if "Nigiri" in card:
            if "Egg" in card:
                val = 1
            elif "Salmon" in card:
                val = 2
            elif "Squid" in card:
                val = 3
            if wasabi_count > 0:
                wasabi_count = wasabi_count - 1
                score = score + (val * 3)
            else:
                assert wasabi_count == 0
                score = score + val
    return score


dumpling_scores = [0, 1, 3, 6, 10, 15, 15, 15, 15, 15, 15, 15, 15]
# TODO: optimize if I have time
def calculate_score(cards):
    counts = [Counter(card) for card in cards]
    dumpling_values = np.array([dumpling_scores[count['Dumpling']] for count in counts])
    tempura_values = np.array([5 * (count['Tempura'] // 2) for count in counts])
    sashimi_values = np.array([10 * (count['Sashimi'] // 3) for count in counts])
    maki_counts = [(3 * count['Maki 3'] + 2 * count['Maki 2'] + 1 * count['Maki 1']) for count in counts]
    maki_values = np.array(calculate_maki(maki_counts))
    nigiri_values = np.array([calculate_nigiri(card) for card in cards])
    return nigiri_values + maki_values + sashimi_values +tempura_values + dumpling_values

def calculate_pudding_score(puddings):
    n = len(puddings)
    mx = max(puddings)
    mn = min(puddings)
    if mx == mn:
        return [0] * n
    num_max = puddings.count(mx)
    num_min = puddings.count(mn)
    output = [0] * n
    for i in range(n):
        if puddings[i] == mx:
            output[i] = 6 // num_max
        elif puddings[i] == mn:
            output[i] = (6 // num_min) * -1
        else:
            output[i] = 0
    if n == 2:
        output = [max([out, 0]) for out in output]
    return output

class Game:

    def __init__(self, n_players, decision_functions = None):

        self.deck = Deck(cards = copy.deepcopy(default_cards_list))
        self.hand_size = 12 - n_players
        self.scores = [0] * n_players
        self.puddings = [0] * n_players
        self.n_players = n_players
        self.hands = deque([None] * n_players)
        self.kept_cards = [None] * n_players
        self.decision_functions = decision_functions
        if self.decision_functions is None:
            self.decision_functions = [default_decision_function, default_decision_function, default_decision_function]
        assert len(decision_functions) == n_players
        assert callable(decision_functions[0])  # asserts for sanity check

    def play_round(self):
        for i in range(self.n_players):
            self.hands[i] = self.deck.draw(self.hand_size)
            self.hands[i].sort()
            self.kept_cards[i] = []
        for j in range(self.hand_size):
            card_choices = [decision(self.hands, self.kept_cards, self.scores, self.puddings, i)
                    for i, decision in enumerate(self.decision_functions)]
            for i in range(self.n_players):
                choice = card_choices[i]
                for j in range(len(choice)):
                    self.kept_cards[i].append(choice[j])
                    self.hands[i].remove(choice[j])
                if len(choice) == 2:
                    assert 'Chopsticks' in self.kept_cards[i]
                    self.hands[i].append('Chopsticks')
                    self.kept_cards[i].remove('Chopsticks')
                self.hands[i].sort()
            self.hands.rotate(1)  # pass hand to adjacent player

        round_scores = calculate_score(self.kept_cards)
        for i in range(self.n_players):
            self.scores[i] = self.scores[i] + round_scores[i]
            self.puddings[i] = self.puddings[i] + self.kept_cards[i].count('Pudding')

    def play(self, n_rounds = 3):
        for _ in range(n_rounds):
            self.play_round()

        print("scores are: ", self.scores)
        print("puddings are: ", self.puddings)
        pudding_scores = calculate_pudding_score(self.puddings)
        print("pudding scores are: ", pudding_scores)
        print("total scores are: ", [a + b for a, b in zip(pudding_scores, self.scores)])

# g = Game(2, [default_decision_function, default_decision_function])
#
# g.play(3)
#
# g_human = Game(2, [human_decision_function, default_decision_function])
#
# g_human.play(1)
