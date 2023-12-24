# Cart Pole
[link to gymnasium doc](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

Action space:
- 0 (move cart to left)
- 1 (move cart to right)

Observation space:
- cart position (between -4.8 and 4.8)
- cart velocity (between -inf and inf)
- pole angle in radians (between -0.418 and 0.418 rad)
- pole angular velocity (between -inf and inf)

Reward: 1 for every step taken

starting state & termination:
- ALL observations are assigned a uniformly random value in (-0.05, 0.05)
- the episode terminates if the cart leaves the (-2.4, 2.4) range
- the episode terminates if the pole angle is not in the range (-.2095, .2095) (or ±12°)
- episode terminates if its length is greater than 500