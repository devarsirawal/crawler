# Crawler

Testing environment for simulation of magnetic wheeled crawler in IsaacGym

To run, execute:
```
python test_crawler.py
```

Command line arguments include:
```
--vertical: Set ground plane vertical, normal to the x-axis
--use_sphere: Use crawler model with a sphere as the caster wheel instead of swivel mechanism
--use_capsule: Replace cylinders with capsules
--damping: Set the damping of the front left/right wheels
--local: Apply force of magnetism in local frame (otherwise env frame)
--force: Apply acceleration for force of magnetism (N/m)
--velocity: Apply velocity to front left/right wheels (rad/s)
--max_plot_time: Iterations to plot
```

URDF files are included in `crawler_description/`. The model only contains wheels and a collision box for the body.
