This is an electromagnetic simulation of a nanotip that uses the MEEP FTDT simulation tools built for Python.  It's primary purpose is to determine the electromagnetic fields surrounding the nanotip given a CW excitation.  Important information regarding the simulation are listed below:

 - Due to the differences in scale, it applies variable resolution s.t. the tip is at much higher resolution sampling than the exterior of the tip.

 - The tip is set with a single dielectric constant (by n + jk) with a CW source.

 - The CW source is turned on at time 0 and runs for the specified duration set by the user as the nunber of cycles (or waves)

 - There are PMLs that surround the simulation cell, and periodic boundary conditions are used in the y-direction so that angled waves are supported.

 - The source comes in at a grazing angle set by theta (in degrees) by the user. To achieve this, it places a sinusoidal phase variation on the source along the y-direction.  The source area is in the xy-plane and the waves propagate in the z-direction with a slight downward angle in y.

 - The tip is made by a cylinder, with a cone on top, capped by a sphere of specified radius of curvature.  