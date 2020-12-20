import json
import numpy as np
import pandas as pd
import re
import itertools
import glob

class myClass (object):
  def __init__(self, f, data):
    self.file_name = f
    self.data = data
  
  def __lt__(self, other):
    return self.file_name < other.file_name

data_list = []

for f in glob.glob('log*'):

  with open(f) as x:
    data_list.append(myClass(f , json.load(x)))

columns_header = ["4x4", "5x5", "6x6", "7x7", "8x8"]

row_header_1 = ["60", "70", "72", "74", "76", "78", "80", "90"]
row_header_2 = ["avg outcome", "std outcome", "grabbed gold","percentage grabbed gold", "avg with gold", "std with gold", "home without gold", "percentage home without gold", "avg without gold", "std without gold", "died", "percentage died", "avg died", "std died"]

row_header_index1 = list(itertools.chain.from_iterable(itertools.repeat(x, len(row_header_2)) for x in row_header_1))
row_header_index2 = row_header_2 * len(row_header_1)

dataframe_dict = {'threshold': row_header_index1, 'stat_data': row_header_index2,
        '4x4': ["a" for i in range(len(row_header_1)*len(row_header_2))],
        '5x5': ["a" for i in range(len(row_header_1)*len(row_header_2))],
        '6x6': ["a" for i in range(len(row_header_1)*len(row_header_2))],
        '7x7': ["a" for i in range(len(row_header_1)*len(row_header_2))],
        '8x8': ["a" for i in range(len(row_header_1)*len(row_header_2))]
         }

dataframe = pd.DataFrame(dataframe_dict)
dataframe = dataframe.set_index(['threshold', 'stat_data'])

for data_entry in sorted(data_list):
  # print("*" * 30)
  data = data_entry.data
  filename = data_entry.file_name
  # print(filename)
  outcomes = []
  alives = []
  outcomes_alive_with_gold = []
  outcomes_alive_without_gold = []
  outcomes_died = []
  for i in range (len(data)):

    outcome = data[i]['outcome']
    alive = data[i]['alive']

    outcomes.append(outcome)
    alives.append(alive)

    if data[i]['alive']:
      if "GRAB" in data[i]['moves']:
        outcomes_alive_with_gold.append(outcome)
      else:
        outcomes_alive_without_gold.append(outcome)
    else:
      outcomes_died.append(outcome)

  o = np.array(outcomes)
  o_gold = np.array(outcomes_alive_with_gold)
  o_not_gold = np.array(outcomes_alive_without_gold)
  o_died = np.array(outcomes_died)

  std = np.std(o)
  std_gold = np.std(o_gold)
  std_not_gold = np.std(o_not_gold)
  std_died = np.std(o_died)

  avg_outcomes = np.mean(o)
  avg_gold = np.mean(o_gold)
  avg_not_gold = np.mean(o_not_gold)
  avg_died = np.mean(o_died)

  n_gold = len(o_gold)
  perc_gold = (n_gold/len(data))*100
    
  n_no_gold = len(o_not_gold)
  perc_no_gold = (n_no_gold/len(data))*100
    
  n_died = len(o_died)
  perc_died = (n_died/len(data))*100

  # print("std = %f" %std)
  # print("avg outcome = %f" %avg_outcomes)
  # print("died = %i" %died)
  # print("percentage died = %f" %perc_died)
  # #print("min outcome = %i" %min_out)
  # #print("max outcome = %i" %max_out)
  # print("std alive with gold = %f" %std_gold)
  # print("avg outcome alive with gold = %f" %avg_gold)
  # print("std alive without gold = %f" %std_not_gold)
  # print("avg outcome alive without gold = %f" %avg_not_gold)
  # print("std died = %f" %std_died)
  # print("avg outcome died = %f" %avg_died)

  dataframe.loc[(data_entry.file_name[8:10], "avg outcome"), [data_entry.file_name[4:7]]] = avg_outcomes
  dataframe.loc[(data_entry.file_name[8:10], "std outcome"), [data_entry.file_name[4:7]]] = std
  dataframe.loc[(data_entry.file_name[8:10], "grabbed gold"), [data_entry.file_name[4:7]]] = n_gold
  dataframe.loc[(data_entry.file_name[8:10], "percentage grabbed gold"), [data_entry.file_name[4:7]]] = perc_gold
  dataframe.loc[(data_entry.file_name[8:10], "avg with gold"), [data_entry.file_name[4:7]]] = avg_gold
  dataframe.loc[(data_entry.file_name[8:10], "std with gold"), [data_entry.file_name[4:7]]] = std_gold
  dataframe.loc[(data_entry.file_name[8:10], "home without gold"), [data_entry.file_name[4:7]]] = n_no_gold
  dataframe.loc[(data_entry.file_name[8:10], "percentage home without gold"), [data_entry.file_name[4:7]]] = perc_no_gold
  dataframe.loc[(data_entry.file_name[8:10], "avg without gold"), [data_entry.file_name[4:7]]] = avg_not_gold
  dataframe.loc[(data_entry.file_name[8:10], "std without gold"), [data_entry.file_name[4:7]]] = perc_no_gold
  dataframe.loc[(data_entry.file_name[8:10], "died"), [data_entry.file_name[4:7]]] = n_died
  dataframe.loc[(data_entry.file_name[8:10], "percentage died"), [data_entry.file_name[4:7]]] = perc_died
  dataframe.loc[(data_entry.file_name[8:10], "avg died"), [data_entry.file_name[4:7]]] = avg_died
  dataframe.loc[(data_entry.file_name[8:10], "std died"), [data_entry.file_name[4:7]]] = std_died