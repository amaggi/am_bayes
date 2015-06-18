import numpy as np

n_floors = 6
my_floor = 1  # american floor - 1
n_elev = 2

npts_mc = 10000

speed = 0.5  # floors / sec
dt = 0.1 # time step
n_cycles = 10
npts = np.int((n_floors/(speed*dt)))*n_cycles


def time_to_my_floor(position, dir_up):

    if position <= my_floor and dir_up:
        t = (my_floor - position) / speed
        up = True
    elif position > my_floor and dir_up:
        t = (n_floors - position + n_floors - my_floor) / speed
        up = False
    elif position >= my_floor and not dir_up:
        t = (position - my_floor) / speed
        up = False
    else:
        t = (position + my_floor) / speed
        up = True

    return t, up
    

np.random.seed()

# set up the monte carlo loop
n_down = 0
wait_time = 0
times = np.empty(n_elev, dtype=float)
dir_up = np.empty(n_elev, dtype=bool)
for i in xrange(npts_mc):
    # get a random position and direction for the elevators
    pos = np.random.rand(n_elev)*n_floors
    up = np.random.rand(n_elev) < 0.5 
    # find the first elevator to arrive
    for ie in xrange(n_elev):
        times[ie], dir_up[ie] = time_to_my_floor(pos[ie], up[ie])
    first_elevator = np.argmin(times)
    wait_time = wait_time + times[first_elevator]
    # if it is going down, add it to the count
    if not dir_up[first_elevator]:
        n_down = n_down + 1
    
print("Average time to wait for first elevator = %.2f seconds" % (wait_time/npts_mc))
print("Probability first elevator going down = %.2f" % (float(n_down)/npts_mc))
