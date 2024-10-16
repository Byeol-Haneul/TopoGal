# SPECIFY MACHINE TO USE

MACHINE = "HAPPINESS" #"RUSTY"

if MACHINE == "HAPPINESS":
    BASE_DIR = "/data2/jylee/topology/"
    DATA_DIR = BASE_DIR + "/IllustrisTNG/combinatorial/"
    RESULT_DIR = BASE_DIR + "/IllustrisTNG/combinatorial/results/"
elif MACHINE=="RUSTY":
    BASE_DIR = "/mnt/home/jlee2/ceph/topology/"
    DATA_DIR = "/mnt/home/jlee2/data_dir/"
    RESULT_DIR = "/mnt/home/jlee2/results/"
else:
    raise Exception("Invalid Machine")
