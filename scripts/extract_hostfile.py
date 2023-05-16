import os
import json
import torch

userName = "lx"
workPath = "/home/lx/projects/GLM"

if os.path.exists(workPath + "/hostfile.json"):
    with open(workPath + "/hostfile.json") as file:
        hosts = json.load(file)
    master_hosts, slave_hosts = [], []
    for host_info in hosts:
        if host_info["role"] == "master":
            master_hosts.append(host_info["ip"])
        else:
            slave_hosts.append(host_info["ip"])
    with open(workPath + "/hostfile", "w") as output:
        for host in master_hosts:
            output.write(f"{userName}@{host} slots=8\n")
        for host in slave_hosts:
            output.write(f"{userName}@{host} slots=8\n")
    with open(workPath + "/pssh_hosts", "w") as output:
        for host in master_hosts:
            output.write(f"{userName}@{host}\n")
        for host in slave_hosts:
            output.write(f"{userName}@{host}\n")
else:
    gpu_count = torch.cuda.device_count()
    with open(workPath + "/hostfile", "w") as output:
        output.write(userName + "@127.0.0.1 slots=8\n")
