# installation
Install swig: `brew install swig` (needed for gymnasium[box2d])
```
python3 -m venv env
source env/bin/activate
env/bin/python3.11 -m pip install -r requirements.txt
```


# OpenAI RL problems

Recently, I have been reading up on reinforcement learning with Stanford's CS234 RL course. I liked it so much I took notes and put it up on my blog, check it out [here](https://timothydelille.com/notes/stanford-cs234-reinforcement-learning-lecture-2).

## Cartpole
I implemented Monte Carlo sampling and Q learning for the cartpole environment in `cartpole.ipynb`.

![cartpole](cartpole.gif)

The pole is balanced! 🤩

## Car racing
To play with the car racing game: `env/lib/python3.11 env/lib/python3.11/site-packages/gymnasium/envs/box2d/car_racing.py`

# VM

port-forwarding. add this to the gcloud ssh command: `-- -NL 6006:localhost:6006`


setup:
```
sudo apt install python3.11-venv
python3 -m venv
source env/bin/activate
sudo apt-get install swig build-essential python-dev-is-python3
python -m pip install -r requirements.txt
```

set up SSH key on the VM following this: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux

install git: `sudo apt-get install git`