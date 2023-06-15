import random


class Gambling(random):

    def playing_baccarat(self, hand, amount):

        odds = {'banker': 1.95, 'player': 2, 'tie': 9, 'pair': 12}
        playing_result = random.random()

        # banker win
        if playing_result < 0.458597 and hand == 1:
            return amount * odds[hand]

        # player win
        elif 0.458597 <= playing_result < 0.904844 and hand == 2:
            return amount * odds[hand]

        # tie
        elif 0.904844 <= playing_result < 1 and hand == 3:
            return amount * odds[hand]

        # pair
        elif hand == 4 and playing_result < 0.074683:
            return amount * odds[hand]

        return 0