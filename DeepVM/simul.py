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
    a_val = v['a']
    b_val = v['b']
    c_val = v['c']

    factor = K(a_val, b_val, c_val, n)
    return factor

def findOptimalTieringArch(willingness, buffer_size, checkpoint_size, scaling: bool):
    V = [instance for instance in data['instances'] if instance['type'] in ('G', 'P')]
    W = [instance for instance in data['instances'] if instance['type'] not in ('G', 'P')]

    config_list = []

    for v in V:
        for w in W:     
            for n in range(1, MAX_LIMIT):  # MAX_LIMIT는 최대 한계값
                for m in range(1, MAX_LIMIT):
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

    return sorted_configs[0]

def findOptimalSingleAnchorArch(willingness, scaling: bool):
    V = [instance for instance in data['instances'] if instance['type'] in ('G', 'P')]

    config_list = []

    for v in V:
        for n in range(2, MAX_LIMIT):  # MAX_LIMIT는 최대 한계값
            if v['spot_price'] * (n-1) + v['ondemand_price'] <= willingness:
                if scaling:
                    Z = FLOPP(v) *n* ScalingFactor(v, n)
                else:
                    Z = FLOPP(v) *n

                config_list.append((Z, v, n))

    if len(config_list) == 0:
       return None
    
    sorted_configs = sorted(config_list, key=lambda x: x[0], reverse=True)

    return sorted_configs[0]


def cost_first(pw):
    max_n = 0
    selected_instance = None

    for instance in data['instances']:
        if instance['type'] == 'G':
            if pw >= instance['ondemand_price'] + instance['spot_price']:
                n_spot = int((pw - instance['ondemand_price']) // instance['spot_price'])
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
        if pw >= instance['ondemand_price'] + instance['spot_price']:
            n_spot = max(int((pw - instance['ondemand_price']) // instance['spot_price']), 1)
            total_cost = n_spot * instance['spot_price'] + instance['ondemand_price']

            if total_cost <= pw:
                return instance, n_spot + 1

    return None

def deepvm_noscale(pw):
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
    
def deepvm(pw):
    tier = findOptimalTieringArch(pw, bf, ckpsize, True)
    sa = findOptimalSingleAnchorArch(pw, True)

    if tier is None and sa is None:
        return None
    elif tier is None:
        return sa
    elif sa is None:
        return tier
    else:
        return tier if tier[0] > sa[0] else sa

def performance_eval(v, n):
    perf = v['flops']  * ScalingFactor(v, n) * n
    if perf<0:
       print(v, n)
    return perf

def load_data(filename):
  with open(filename, "r", encoding="utf-8") as file:
    return json.load(file)

data = load_data("simul_data.json")
bf = 2
ckpsize = 0.5
MAX_LIMIT = 256
pw_array = np.arange(0, 10.1, 0.1)

results = {
    'pw': pw_array,
    'cost_first': [],
    'performance_first': [],
    'deepvm_noscale': [],
    'deepvm': []
}

start_time = time.time()
iterations = len(pw_array)
for i, pw in enumerate(pw_array):
    progress = (i + 1) / iterations
    bar_length = 20
    block = int(round(bar_length * progress))
    if i == 0:
        text = "\r[{0}] {1:.2f}% ({2}/{3})".format("#" * block + "-" * (bar_length - block), progress * 100, i+1, iterations)
    else:
        t = ((time.time()-start_time)/i)*(iterations-i)
        text = "\r[{0}] {1:.2f}% ({2}/{3}) {4:.0f}m {5:.0f}s".format("#" * block + "-" * (bar_length - block), progress * 100, i, iterations, t//60, t%60)
    sys.stdout.write("\033[K")
    print(text, end='')
    sys.stdout.flush()

    cost_first_config = cost_first(pw)
    performance_first_config = performance_first(pw)
    deepvm_noscale_config = deepvm_noscale(pw)
    deepvm_config = deepvm(pw)

    # print(cost_first_config)
    # print(performance_first_config)
    # print(deepvm_noscale_config)
    # print(deepvm_config)

    if cost_first_config is not None:
       results['cost_first'].append(performance_eval(cost_first_config[0], cost_first_config[1]))
    
    if performance_first_config is not None:
       results['performance_first'].append(performance_eval(performance_first_config[0], performance_first_config[1]))
    
    if deepvm_noscale_config is not None:
        if len(deepvm_noscale_config) == 3:
            results['deepvm_noscale'].append(performance_eval(deepvm_noscale_config[1], deepvm_noscale_config[2]))
        else:
            results['deepvm_noscale'].append(performance_eval(deepvm_noscale_config[1], deepvm_noscale_config[3]))
    
    if deepvm_config is not None:
        if len(deepvm_config) == 3:
            results['deepvm'].append(performance_eval(deepvm_config[1], deepvm_config[2]))
        else:
            results['deepvm'].append(performance_eval(deepvm_config[1], deepvm_config[3]))
    end_time = time.time()

# print(results)

with open('simul_results.pkl', 'wb') as f:
    pickle.dump(results, f)