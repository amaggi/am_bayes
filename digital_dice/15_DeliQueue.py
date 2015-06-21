import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

nsamp = 10000

def setup_queue(lam, mu, tol_time, time_limit):

    # set up arrival intervals and serving times as poisson processes
    arrival_intervals = poisson.rvs(lam, size=nsamp)
    serving_times = poisson.rvs(mu, size=nsamp)
    tolerance_times = poisson.rvs(tol_time, size=nsamp)

    # calculate arrival times
    arrival_times = np.empty(nsamp, dtype=float)
    arrival_times[0] = arrival_intervals[0]
    for i in xrange(nsamp-1):
        arrival_times[i+1] = arrival_times[i] + arrival_intervals[i+1]

    # find number of customers arriving before closing time
    n_customers = np.searchsorted(arrival_times, time_limit) 

    # cut the arrays down to size
    arrival_times = np.resize(arrival_times, n_customers)
    serving_times = np.resize(serving_times, n_customers)

    return arrival_times, serving_times, tolerance_times


def run_day(n_servers, lam, mu, tol_time, time_limit, time_step,
            outside=False):

    # set up the queues
    arrival_times, serving_times, tolerance_times =\
        setup_queue(lam, mu, tol_time, time_limit)
    n_cust = len(arrival_times)
    n_times = np.int(float(time_limit)/time_step)

    # set up arrays for bookkeeping
    wait_times = np.zeros(n_cust, dtype=float)
    queue_length = np.zeros(n_times*5, dtype=int)
    clerk_service_time_remaining = np.zeros(n_servers, dtype=float)
    overtime = 0

    # initialize
    queue = []
    t = 0
    n_steps = 0
    idle_time = 0
    icust = -1
    n_unserved = 0

    while t < time_limit or len(queue) > 0: 

        # continue service of customers being processed
        clerk_service_time_remaining = clerk_service_time_remaining - time_step

        # if a clerk is available and queue is not empty
        clerk0 = np.argmin(clerk_service_time_remaining)
        if clerk_service_time_remaining[clerk0] <= 0. and len(queue) > 0:
            # start serving the first person in the queue
            icust = queue.pop(0)
            clerk_service_time_remaining[clerk0] = serving_times[icust]

        # check if a new customer has arrived and add to the end of the queue
        next_cust =  np.searchsorted(arrival_times, t) -1
        if next_cust > icust:
            #import pdb; pdb.set_trace()
            if len(queue) == 0:
                queue.append(next_cust)
            elif next_cust > queue[-1]:
                queue.append(next_cust)

        # measure the length of the queue
        queue_length[n_steps] = len(queue)

        # add idle time for clerks
        for ic in xrange(n_servers):
            if clerk_service_time_remaining[ic] <= 0:
                idle_time = idle_time + time_step

        # add waiting time for all customers in the queue
        for icust in queue:
            wait_times[icust] = wait_times[icust] + time_step
            # if wait times are too long, some custermers may decide not to
            # stay
            if wait_times[icust] > tolerance_times[icust]:
                queue.remove(icust)
                n_unserved = n_unserved + 1

        # if we are still serving customers after closing time, add to overtime
        if t > time_limit:
            overtime = overtime + time_step
        # add to time
        t = t + time_step
        n_steps = n_steps + 1
            
    # cut the queue length array down to size
    queue_length.resize(n_steps)

    wait_time_mean = np.mean(wait_times)
    wait_time_max = np.max(wait_times)
    queue_length_mean = np.mean(queue_length)
    queue_length_max = np.max(queue_length)

    if outside:
        return len(arrival_times), np.mean(wait_times+serving_times), overtime
    else:
        return len(arrival_times), wait_time_mean, wait_time_max,\
               queue_length_mean, queue_length_max, idle_time, overtime,\
               n_unserved 


if __name__ == '__main__':

    n_servers = 1
    time_limit = 36000 # seconds in a 10-hour day

    arriving_cust_per_hour = 3
    serving_time_minutes = 25

    tol_time_minutes = 60

    n_iterations = 10

    # get average times in seconds
    lam = float(3600) / arriving_cust_per_hour
    mu = serving_time_minutes * 60
    tol_time = tol_time_minutes * 60

    results = np.empty((n_iterations, 8))

    # run simulations
    for i in xrange(n_iterations):
        results[i, :] = run_day(n_servers, lam, mu, tol_time, time_limit, 10.0)
    results_mean = np.mean(results, axis=0)
    results_std = np.std(results, axis=0)

    # print results
    print("-------------")
    print("Input : %.1f customers per hour"% arriving_cust_per_hour)
    print("Input : Average wait tolerance time %.1f min" % (tol_time/60.))
    print("Input : Average serving time  %.1f min"% (mu/60.))
    print("Input : %d servers"% n_servers)
    print("-------------")
    print("Total customers served : %.1f +- %.1f" % (results_mean[0],
                                                     results_std[0]))
    print("Average wait time : %.1f +- %.1f min" % (results_mean[1]/60.,
                                                results_std[1]/60.))
    print("Maximum wait time : %.1f +- %.1f min" % (results_mean[2]/60.,
                                                results_std[2/60.]))
    print("Average queue length : %.1f +- %.1f" % (results_mean[3],
                                                   results_std[3]))
    print("Maximum queue length : %.1f +- %.1f" % (results_mean[4],
                                                   results_std[4]))
    print("Total idle time : %.1f +- %.1f min" % (results_mean[5]/60.,
                                                results_std[5]/60.))
    print("Overtime : %.1f +- %.1f min" % (results_mean[6]/60.,
                                           results_std[6]/60.))
    print("Unserved customers : %.1f +- %.1f" % (results_mean[7],
                                                     results_std[7]))


