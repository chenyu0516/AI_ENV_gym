import random


class Gambling:

    def playing_baccarat(self, hand, percentage, value):

        odds = [0.95, 1, 8, 11]
        playing_result = random.random()

        # banker win
        if playing_result < 0.458597 and hand == 1:
            return percentage * value * odds[hand]

        # player win
        elif 0.458597 <= playing_result < 0.904844 and hand == 2:
            return percentage * value * odds[hand]

        # tie
        elif 0.904844 <= playing_result < 1 and hand == 3:
            return percentage * value * odds[hand]

        # pair
        elif hand == 4 and playing_result < 0.074683:
            return percentage * value * odds[hand]

        return -percentage * value