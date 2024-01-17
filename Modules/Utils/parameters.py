"""Module providing utils classes."""
class OptionParam:
    """
Define option.
s_0 = current stock price
k = Strike price
t = time to maturity
v0 = (optional) spot variance
exercise = European or American


"""

    def __init__(self, s_0, k, t_mat, v_0, payoff, exercise):
        self.s_0 = s_0
        self.v_0 = v_0
        self.k = k
        self.t_mat = t_mat

        if exercise.upper == "EUROPEAN" or exercise.upper == "AMERICAN":
            self.exercise = exercise
        else:
            raise ValueError("invalid type.")
        if payoff.upper == "CALL" or payoff.upper == "PUT":
            self.payoff = payoff
        else:
            raise ValueError("invalid type.")
        