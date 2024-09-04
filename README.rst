===================
Chasing Targets Gym
===================

|version| |python| |license| |codestyle|

.. |version| image:: https://img.shields.io/pypi/v/chasing-targets-gym
    :target: https://pypi.org/project/chasing-targets-gym/
    :alt: PyPI - Package Version
.. |python| image:: https://img.shields.io/pypi/pyversions/chasing-targets-gym
    :target: https://pypi.org/project/chasing-targets-gym/
    :alt: PyPI - Python Version
.. |license| image:: https://img.shields.io/pypi/l/chasing-targets-gym
    :target: https://github.com/5had3z/chasing-targets-gym/blob/main/LICENSE
    :alt: PyPI - License
.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


Introduction
------------

This is a simple gym environment that sets up a set of robots and targets for them to chase.
These targets are dumb, they simply move at a constant speed and bounce off the "limits" of 
the simulation environment. The intention is that the robots will chase after these targets,
and switch to a new target after catching their current one. The targets are "transparent" and
robots are free to ignore "avoiding them", the intention is that they avoid each other. An
example of a simulation with robot controller is shown below.

.. image:: misc/example_sim.gif


Usage
-----

Since this uses the gymnasium, you can spin an environment up same as any other env, and you can use our optimized planner. A script is included that shows of this planner and environment when you install this library ```chasing-targets-example --max-step=500```.

.. code:: python
    
    from gymnasium import Env, make
    from chasing_targets_gym.planner import Planner

    env: Env = make(
        "ChasingTargets-v0",
        render_mode="human",
        n_robots=10,
        n_targets=3,
        robot_radius=0.1,
        max_velocity=0.5,
        barrier_velocity_range=0.5,
        max_episode_steps=1000,
    )

    planner = Planner(
        env.get_wrapper_attr("robot_radius"),
        env.get_wrapper_attr("dt"),
        env.get_wrapper_attr("max_velocity"),
    )



Installation
------------

Either you can clone and pip install the source, or you can install via pypi.

.. code:: bash

    git clone https://github.com/5had3z/chasing-targets-gym && cd chasing-targets-gym && pip3 install -e .

Otherwise install pypi package

.. code:: bash

    pip3 install chasing-targets-gym


Some Credit
-----------

I was pointed to a basic environment `here <https://github.com/riiswa/planning-multi-robot-gym>`_ but it didn't
really match what I wanted, so I made my own based off this.
