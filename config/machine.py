MACHINE = "RUSTY" #"HAPPINESS"
TYPE = "BISPECTRUM"   #"CAMELS" #"BISPECTRUM", #"fR"
SUBGRID = "IllustrisTNG" #"IllustrisTNG", "SIMBA", SIMBA is only for testing robustness.

if MACHINE == "HAPPINESS":
    BASE_DIR = "/data2/jylee/topology/"
    DATA_DIR = BASE_DIR + "/IllustrisTNG/combinatorial/"
    RESULT_DIR = BASE_DIR + "/IllustrisTNG/combinatorial/results/"
elif MACHINE=="RUSTY":
    if TYPE == "BISPECTRUM":
        BASE_DIR = "/mnt/home/jlee2/bispectrum/"
        DATA_DIR = "/mnt/home/jlee2/bispectrum/"
        RESULT_DIR = "/mnt/home/jlee2/bispectrum/results/"
    elif TYPE == "CAMELS":
        BASE_DIR = f"/mnt/home/jlee2/ceph/topology/{SUBGRID}/"
        DATA_DIR = f"/mnt/home/jlee2/data_dir/{SUBGRID}/"
        RESULT_DIR = f"/mnt/home/jlee2/results/{SUBGRID}/"
    else:
        raise Exception("Invalid Simulation Suite")
else:
    raise Exception("Invalid Machine")

LABEL_FILENAME = BASE_DIR + "new/latin_hypercube_params.txt" if TYPE == "BISPECTRUM" else BASE_DIR + f"CosmoAstroSeed_{SUBGRID}_L25n256_LH.txt"

if TYPE == "BISPECTRUM":
    CATALOG_SIZE = 2000
else:
    if SUBGRID == "IllustrisTNG": 
        CATALOG_SIZE = 1000
    else:
        CATALOG_SIZE = 100