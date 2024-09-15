# SPECIFY MACHINE TO USE

MACHINE = "HAPPINESS" #"RUSTY" # 

if MACHINE == "HAPPINESS":
    BASE_DIR = "/data2/jylee/topology/"
elif MACHINE=="RUSTY":
    BASE_DIR = "/mnt/home/jlee2/ceph/topology/"
else:
    raise Exception("Invalid")