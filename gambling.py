import random


class Gambling:

    def playing_baccarat(self, index, value):

        percentage = (index % 10)+1 / 1000
        hand = int(index/10)
        odds = [0.95, 1, 8, 11]
        playing_result = random.random()
        print({'percentage': percentage, 'hand': hand})
        if value < 0:
            return 0
        # banker win
        elif playing_result < 0.458597 and hand == 0:
            return percentage * value * odds[hand]

        # player win
        elif 0.458597 <= playing_result < 0.904844 and hand == 1:
            return percentage * value * odds[hand]

        # tie
        elif 0.904844 <= playing_result < 1 and hand == 2:
            return percentage * value * odds[hand]

        # pair
        elif hand == 3 and playing_result < 0.074683:
            return percentage * value * odds[hand]

        # no action
        elif index == 40:
            return 0

        return -percentage * value