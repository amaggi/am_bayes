import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from copy import deepcopy

input_fname = 'incorporations.txt'


def read_input():

    # read the input data from the file
    names = list(['DAY1', 'MON1', 'YEAR1', 'DAY2', 'MON2', 'YEAR2', 'COMP',
                  'COLOR', 'SCHOOL'])
    pd_data = pd.read_table(input_fname, sep='\s+', names=names, header=0,
                            na_values=['-'])

    # transform the dates
    names_date1 = list(['DAY1', 'MON1', 'YEAR1'])
    names_date2 = list(['DAY2', 'MON2', 'YEAR2'])
    date1 = pd_data[names_date1].values
    date2 = pd_data[names_date2].values
    nprom, nd = date1.shape
    d1 = np.empty(nprom, dtype=object)
    d2 = np.empty(nprom, dtype=object)
    comp = pd_data['COMP'].values
    for i in xrange(nprom):
        d1[i] = date(np.int(date1[i, 2])+2000, np.int(date1[i, 1]),
                        np.int(date1[i, 0]))
        d2[i] = date(np.int(date2[i, 2])+2000, np.int(date2[i, 1]),
                        np.int(date2[i, 0]))

    data = np.vstack((d1, d2, comp, pd_data['SCHOOL'].values)).T

    # fix bugs in input data
    for i in xrange(nprom):
        prom = data[i, :]
        # Montlucon 7 orange overlap
        if prom[0]==date(2013, 12, 10) and prom[2]==7:
            data[i, 0] = prom[0] - timedelta(14)
            data[i, 1] = prom[1] - timedelta(14)

    return data


def plot_stats(data):
    
    n, nd = data.shape
    duration = np.array([(data[i, 1] - data[i, 0]).days for i in xrange(n)])
    end_day = np.array([data[i, 1].weekday() for i in xrange(n)])
    school = data[:, 3]


    # Length of training
    bins = np.arange(269, 275)+0.5
    plt.figure(facecolor='w')
    plt.hist(duration, bins=bins)
    plt.xlabel('Duree ecole (jours)')
    plt.ylabel('Nombre de formations')
    plt.title('Duree des formations')

    # Length of training
    bins = np.arange(269, 275)+0.5
    dur_chat = duration[school=='CHATEAULIN']
    dur_mont = duration[school=='MONTLUCON']
    dur_chau = duration[school=='CHAUMONT']
    dur_tull = duration[school=='TULLE']
    plt.figure(facecolor='w')
    plt.hist([dur_chat, dur_mont, dur_chau, dur_tull], bins=bins,
             label=['Chateaulin', 'Montlucon', 'Chaumont', 'Tulle'])
    plt.xlabel('Duree ecole (jours)')
    plt.ylabel('Nombre de formations')
    plt.title('Duree des formations')
    plt.legend()

    # Day of week for end of training
    bins = np.arange(0, 8)-0.5
    plt.figure(facecolor='w')
    plt.hist(end_day, bins=bins)
    plt.xlabel('Jour sortie')
    plt.ylabel('Nombre de formations')
    plt.title('Jours de sortie des formations')

    # Days between successive groups
    idle_days = get_idle_days(data)
    idle_median = np.median(np.concatenate(idle_days.values()))
    idle_low =  np.percentile(np.concatenate(idle_days.values()), 5)
    idle_high = np.percentile(np.concatenate(idle_days.values()), 95)
    plt.figure(facecolor='w')
    bins = np.arange(58)*7
    plt.hist([idle_days[key] for key in idle_days.keys()], bins=bins,
             label = [key for key in idle_days.keys()], stacked=True)
    plt.xlabel('Jours entre promotions')
    plt.ylabel('Nombre de formations')
    plt.title('Jours entre les formations : %.1f - %.1f - %.1f' %
              (idle_low, idle_median, idle_high))
    plt.legend()

def get_idle_days(data):
    idle_days = {}
    data_by_school = split_data_by_school(data)
    for key, value in data_by_school.iteritems():
        idle_days[key] = []
        data_by_company =  split_data_by_company(value)
        for key2, value2 in data_by_company.iteritems():
            data_sorted = np.sort(value2, axis=0)
            n, nd = data_sorted.shape
            for i in xrange(n-1):
                n_idle_days = (data_sorted[i+1, 0]-data_sorted[i, 1]).days
                idle_days[key].append(n_idle_days)

    return idle_days
 
def state_at_time(data, t):

    n, nd = data.shape

    state_dict = {}
    # set up dictonary with school names
    schools = np.unique(data[:, 3])
    for s in schools:
        state_dict[s] = []

    # loop over all data and add current groups to dictionary
    for i in xrange(n):
        start_time = data[i, 0]
        end_time = data[i, 1]
        comp = data[i, 2]
        school = data[i, 3]
        if t >= start_time and t <= end_time:
            state_dict[school].append(comp)
        
    return state_dict

def find_nan_proms(data):

    n, nd = data.shape

    comp = data[:, 2]
    nan_indexes = []
    for i in xrange(n):
        if np.isnan(comp[i]):
            nan_indexes.append(i)
    nan_data = data[nan_indexes, :]

#     # make up non-nan indexes
# 
#     ok_indexes = range(n)
#     for i in nan_indexes:
#         ok_indexes.remove(i)
# 
#     data_dict = split_data_by_school(data[ok_indexes, :])
#     for key, value in data_dict.iteritems():
#         print key, np.unique(value[:, 2])
# 
#     n_nan, nd = nan_data.shape
#     for index in xrange(n_nan):
#         school = nan_data[index, 3]
#         state_start = state_at_time(data_dict[school], nan_data[index, 0])
#         state_end = state_at_time(data_dict[school], nan_data[index, 1])
#         print nan_data[index, :]
#         print state_start
#         print state_end
        

    return nan_data
        

def evaluate_consistency(data):

    non_consistent = {}

    # split data by school
    data_by_school = split_data_by_school(data)
    for key, value in data_by_school.iteritems():
        # initialize inconsistency counter
        non_consistent[key] = 0
        # re-split data by company
        data_by_company =  split_data_by_company(value)
        for key2, value2 in data_by_company.iteritems():
            data_sorted = np.sort(value2, axis=0)
            n, nd = data_sorted.shape
            # loop over all proms by company and
            for i in xrange(n):
                for j in xrange(n):
                    if not j==i:
                        ok = compatible(data_sorted[j, :], data_sorted[i, :])
                        if not ok:
                            non_consistent[key] = non_consistent[key] + 1

    return non_consistent


def misfit_score(data):
    non_consistent = evaluate_consistency(data)
    nan_data = find_nan_proms(data)
    n, nd = nan_data.shape
    score = np.sum(non_consistent.values()) + n
    
    return score

def compatible(prom1, prom2):
    if prom1[0] >= prom2[0] and prom1[0] <= prom2[1]:
        return False
    elif prom1[1] >= prom2[0] and prom1[1] <= prom2[1]:
        return False
    else:
        return True


def split_data_by_school(data):

    data_dict = {}
    schools = np.unique(data[:, 3])
    for s in schools : 
        data_dict[s] = data[:, :][data[:, 3]==s]
    
    return data_dict

def split_data_by_company(data):

    data_dict = {}
    comps = np.unique(data[:, 2])
    for c in comps : 
        data_dict[c] = data[:, :][data[:, 2]==c]
    
    return data_dict

if __name__ == '__main__':

    data = read_input()
    #t = date.today()
    #print state_at_time(data, t)
    #print find_nan_proms(data)
    #print evaluate_consistency(data)
    print misfit_score(data)
    plot_stats(data)
    plt.show()
