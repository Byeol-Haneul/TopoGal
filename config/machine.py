# SPECIFY MACHINE TO USE

MACHINE = "RUSTY" #"HAPPINESS"
TYPE = "CAMELS" #"BISPECTRUM" #"CAMELS" #"BISPECTRUM"

if MACHINE == "HAPPINESS":
    BASE_DIR = "/data2/jylee/topology/"
    DATA_DIR = BASE_DIR + "/IllustrisTNG/combinatorial/"
    RESULT_DIR = BASE_DIR + "/IllustrisTNG/combinatorial/results/"
elif MACHINE=="RUSTY":
    BASE_DIR = "/mnt/home/jlee2/bispectrum/" if TYPE == "BISPECTRUM" else "/mnt/home/jlee2/ceph/topology/"
    DATA_DIR = "/mnt/home/jlee2/bispectrum/" if TYPE == "BISPECTRUM" else "/mnt/home/jlee2/data_dir/"
    RESULT_DIR = "/mnt/home/jlee2/bispectrum/results/" if TYPE == "BISPECTRUM" else "/mnt/home/jlee2/results/"
else:
    raise Exception("Invalid Machine")

LABEL_FILENAME = BASE_DIR + "new/latin_hypercube_params.txt" if TYPE == "BISPECTRUM" else BASE_DIR + "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
CATALOG_SIZE = 2000 if TYPE == "BISPECTRUM" else 1000