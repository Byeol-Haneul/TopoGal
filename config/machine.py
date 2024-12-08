MACHINE = "RUSTY" #"HAPPINESS"
TYPE = "CAMELS"       #"CAMELS" #"BISPECTRUM", #"fR"
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
        LABEL_FILENAME = BASE_DIR + "sims/latin_hypercube_params.txt"
    elif TYPE == "CAMELS":
        BASE_DIR = f"/mnt/home/jlee2/camels/{SUBGRID}/"
        DATA_DIR = f"/mnt/home/jlee2/camels/{SUBGRID}/"
        RESULT_DIR = f"/mnt/home/jlee2/camels/{SUBGRID}/results/{SUBGRID}/"
        LABEL_FILENAME = BASE_DIR + f"CosmoAstroSeed_{SUBGRID}_L25n256_LH.txt"
    elif TYPE == "fR":
        BASE_DIR = "/mnt/home/jlee2/fR/"
        DATA_DIR = "/mnt/home/jlee2/fR/"
        RESULT_DIR = "/mnt/home/jlee2/fR/results/"
        LABEL_FILENAME = BASE_DIR + "fR_labels.txt"
    else:
        raise Exception("Invalid Simulation Suite")
else:
    raise Exception("Invalid Machine")

if TYPE == "BISPECTRUM":
    CATALOG_SIZE = 2000
else:
    if SUBGRID == "IllustrisTNG": 
        CATALOG_SIZE = 1000
    else:
        CATALOG_SIZE = 100