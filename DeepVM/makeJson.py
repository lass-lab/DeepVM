import json
import os

def load_data(filename):
  if os.path.exists(filename):
    with open(filename, "r", encoding="utf-8") as file:
      return json.load(file)
  else:
    return {
      "instances": [],
      "available_vcpus": {}
    }

def add_instance(instances):
  name = input("Enter instance name: ")
  type = input("Enter instance type: ")
  ondemand_price = float(input("Enter ondemand price: "))
  spot_price = float(input("Enter spot price: "))
  vCPU = int(input("Enter vCPU: "))
  memory = float(input("Enter memory size: "))
  network_bandwidth = float(input("Enter network bandwidth (Gbps): "))
  
  instance = {
    "name": name,
    "type": type,
    "ondemand_price": ondemand_price,
    "spot_price": spot_price,
    "vCPU": vCPU,
    "memory": memory,
    "network_bandwidth": network_bandwidth
  }
  
  instances.append(instance)

def set_available_vcpus(available_vcpus):
  type_ = input("Enter instance type: ")
  ondemand = int(input("Enter available ondemand vCPUs: "))
  spot = int(input("Enter available spot vCPUs: "))
  
  available_vcpus[type_] = {
    "ondemand": ondemand,
    "spot": spot
  }

def main():
  data = load_data("data.json")
  
  while True:
    print("\n1. Add Instance")
    print("2. Set Available vCPUs")
    print("3. Save to JSON")
    print("4. Exit")
    
    choice = int(input("Enter your choice: "))
    
    if choice == 1:
      add_instance(data["instances"])
    elif choice == 2:
      set_available_vcpus(data["available_vcpus"])
    elif choice == 3:
      with open("data.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
      print("Saved to data.json")
    elif choice == 4:
      break

if __name__ == "__main__":
  main()
