import numpy as np
import json
import pickle
import time
import sys

def S(a,b,c,n):
  return c / (1 + np.exp(-a * (n - b)))

def S_prime(a,b,c,n):
  return c * a * np.exp(-a * (n - b)) / (1 + np.exp(-a * (n - b)))**2

def L(a,b,c,n):
  return S_prime(a,b,c,b) * (n - b) + S(a,b,c,b)

def K(a,b,c,n):
  if n<=b:
    return L(a,b,c,n) / L(a,b,c,n)
  return S(a,b,c,n) / L(a,b,c,n)

def constraint1(v_price, w_price, n, m, willingness):
  return v_price * n + w_price * m <= willingness

def constraint2(w_memory, checkpoint_size, buffer_size):
  return w_memory >= checkpoint_size * buffer_size

def NWSaturationPoint(v, w):
  bw = min(v['network_bandwidth'], w['network_bandwidth'])
  table = {
    0.3: 2,
    1.7: 12,
    5: 16,
    10: 20,
    12.5: 24,
    15: 24,
    25: 28,
    30: 32,
  }
  point = table[bw]
  return point

def constraint3(n, m, v, w):
  return n/m < NWSaturationPoint(v, w)

def FLOPP(v, type='spot'):
  if(type == 'spot'):
    spot_price = v['spot_price']
    flops = v.get('flops', 0)  # defaults to 0 if not provided
    return flops / spot_price * 3600
  else:
    ondemand_price = v['ondemand_price']
    flops = v.get('flops', 0)  # defaults to 0 if not provided
    return flops / ondemand_price * 3600

def ScalingFactor(v, n):
  a_values = {
    "g3s.xlarge": 0.1408559584,
    "g4dn.xlarge": 0.1338806402,
    "g5.xlarge": 0.08533428072
  }
  b_values = {
    "g3s.xlarge": 14.49263334,
    "g4dn.xlarge": 12.87424514,
    "g5.xlarge": 20.06669931
  }
  c_values= {
    "g3s.xlarge": 13.53250952,
    "g4dn.xlarge": 6.176666667,
    "g5.xlarge": 4.962225551
  }

  a_val = a_values[v['name']]
  b_val = b_values[v['name']]
  c_val = c_values[v['name']]

  factor = K(a_val, b_val, c_val, n)
  # if v['name'] == 'g3s.xlarge' and n==32:
  #   print("g3", factor)
  # if v['name'] == 'g4dn.xlarge' and n==32:
  #   print("g4dn", factor)
  # if v['name'] == 'g5.xlarge' and n==32:
  #   print("g5", factor)
  return factor

def findOptimalTieringArch(willingness, buffer_size, checkpoint_size, scaling: bool):
    V = [instance for instance in data['instances'] if instance['type'] in ('G', 'P')]
    W = [instance for instance in data['instances'] if instance['type'] not in ('G', 'P')]

    config_list = []

    for v in V:
        for w in W:     
            n_max = data['available_vcpus'][v['type']]['spot'] // v['vCPU']
            m_max = data['available_vcpus'][w['type']]['ondemand'] // w['vCPU']
            for n in range(1, n_max):
                for m in range(1, m_max):
                    if (
                        constraint1(v['spot_price'], w['ondemand_price'], n, m, willingness) and
                        constraint2(w['memory'], checkpoint_size, buffer_size) and
                        constraint3(n, m, v, w)
                    ):
                        if scaling:
                            Z = FLOPP(v) * ScalingFactor(v, n) *n
                        else:
                            Z = FLOPP(v)*n
                        config_list.append((Z, v, w, n, m))
    if len(config_list) == 0:
       return None

    sorted_configs = sorted(config_list, key=lambda x: x[0], reverse=True)

    return sorted_configs[:3] if scaling else sorted_configs[0]

def findOptimalSingleAnchorArch(willingness, scaling: bool):
    V = [instance for instance in data['instances'] if instance['type'] in ('G', 'P')]

    config_list = []

    for v in V:
        n_max = data['available_vcpus'][v['type']]['spot'] // v['vCPU']
        for n in range(2, n_max):  # MAX_LIMIT는 최대 한계값
            if v['spot_price'] * (n-1) + v['ondemand_price'] <= willingness:
                if scaling:
                    Z = FLOPP(v) *n* ScalingFactor(v, n)
                else:
                    Z = FLOPP(v) *n

                config_list.append((Z, v, n))

    if len(config_list) == 0:
       return None
    
    sorted_configs = sorted(config_list, key=lambda x: x[0], reverse=True)

    return sorted_configs[:3] if scaling else sorted_configs[0]


def cost_first(pw):
    max_n = 0
    selected_instance = None

    for instance in data['instances']:
        if instance['type'] == 'G':
            n_max = data['available_vcpus'][instance['type']]['spot'] // instance['vCPU']
            if pw >= instance['ondemand_price'] + instance['spot_price']:
                n_spot = int((pw - instance['ondemand_price']) // instance['spot_price'])
                n_spot = min(n_max-1, n_spot)
                total_cost = n_spot * instance['spot_price'] + instance['ondemand_price']

                if total_cost <= pw and n_spot + 1 > max_n:
                    max_n = n_spot + 1
                    selected_instance = instance

    if max_n < 2:
       return None

    return selected_instance, max_n

def performance_first(pw):
    sorted_instances = sorted([inst for inst in data['instances'] if inst['type'] == 'G'], 
                              key=lambda x: x['flops'], reverse=True)

    for instance in sorted_instances:
        n_max = data['available_vcpus'][instance['type']]['spot'] // instance['vCPU']
        if pw >= instance['ondemand_price'] + instance['spot_price']:
            n_spot = max(int((pw - instance['ondemand_price']) // instance['spot_price']), 1)
            n_spot = min(n_max-1, n_spot)
            total_cost = n_spot * instance['spot_price'] + instance['ondemand_price']

            if total_cost <= pw:
                return instance, n_spot + 1

    return None

def deepvm_noscale(pw,bf,ckpsize):
    tier = findOptimalTieringArch(pw, bf, ckpsize, False)
    sa = findOptimalSingleAnchorArch(pw, False)

    if tier is None and sa is None:
        return None
    elif tier is None:
        return sa
    elif sa is None:
        return tier
    else:
        return tier if tier[0] > sa[0] else sa
    
def deepvm(pw,bf,ckpsize):
    tier_configs = findOptimalTieringArch(pw, bf, ckpsize, True)
    sa_configs = findOptimalSingleAnchorArch(pw, True)

    if not tier_configs and not sa_configs:
        return None

    combined_configs = (tier_configs or []) + (sa_configs or [])
    sorted_combined_configs = sorted(combined_configs, key=lambda x: x[0], reverse=True)

    return sorted_combined_configs[:3]

def performance_eval(v, n):
    perf = v['flops']  * ScalingFactor(v, n) * n
    if perf<0:
       print(v, n)
    return perf

def load_data(filename):
  with open(filename, "r", encoding="utf-8") as file:
    return json.load(file)

def main(pw, bf, ckpsize):
  global data
  data = load_data("data.json")

  cost_first_config = cost_first(pw)
  performance_first_config = performance_first(pw)
  deepvm_noscale_config = deepvm_noscale(pw,bf,ckpsize)
  deepvm_config = deepvm(pw,bf,ckpsize)

  print('====== Results of DeepVM ======')
  print('========== cost first =========')
  print(cost_first_config)
  print('===============================')
  print('========== perf first =========')
  print(performance_first_config)
  print('===============================')
  print('======= DeepVM no_scale =======')
  print(deepvm_noscale_config)
  print('===============================')
  print('=========== DeepVM ===========')
  print(deepvm_config)
  print('===============================')

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser (description='DeepVM')
  parser.add_argument ('--pw', default=3, type=float, help='pricing willingness')
  parser.add_argument ('--buffer_size', default=0.5, type=float, help='buffer size of a remote node')
  parser.add_argument ('--ckp_file_size', default=2, type=float, help='checkpoint file size of target DL model')
  args = parser.parse_args ()

  main(args.pw, args.buffer_size, args.ckp_file_size)