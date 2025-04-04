# Configuration settings:
# - MACHINE: Specifies the computational environment. Options: "RUSTY", "HAPPINESS".
# - TYPE: Defines the type of simulation dataset.     Options: "Quijote", "CAMELS".
# - SUBGRID: Allows extensibility for different subgrid models in the future. Current option: "IllustrisTNG".

MACHINE = "RUSTY"
TYPE    = "CAMELS_50"
SUBGRID = "IllustrisTNG"

if MACHINE == "HAPPINESS":
    BASE_DIR   = "/data2/jylee/topology/"
    DATA_DIR   = BASE_DIR + "/IllustrisTNG/combinatorial/"
    RESULT_DIR = BASE_DIR + "/IllustrisTNG/combinatorial/results/"
    
elif MACHINE=="RUSTY":
    if TYPE == "Quijote":
        BASE_DIR       = "/mnt/home/jlee2/quijote/"
        DATA_DIR       = "/mnt/home/jlee2/quijote/"
        RESULT_DIR     = "/mnt/home/jlee2/quijote/results/"
        LABEL_FILENAME = BASE_DIR + "sims/latin_hypercube_params.txt"
    elif TYPE == "Quijote_Rockstar":
        BASE_DIR       = "/mnt/home/jlee2/ceph/quijote_rockstar/"
        DATA_DIR       = "/mnt/home/jlee2/ceph/quijote_rockstar/"
        RESULT_DIR     = "/mnt/home/jlee2/ceph/quijote_rockstar/results/"
        LABEL_FILENAME = BASE_DIR + "sims/BSQ_params.txt"
    elif TYPE == "fR":
        BASE_DIR       = "/mnt/home/jlee2/ceph/fR/"
        DATA_DIR       = "/mnt/home/jlee2/ceph/fR/"
        RESULT_DIR     = "/mnt/home/jlee2/ceph/fR/results/"
        LABEL_FILENAME = BASE_DIR + "full_s8_table.dat"
    elif TYPE == "CAMELS":
        BASE_DIR       = f"/mnt/home/jlee2/camels/{SUBGRID}/"
        DATA_DIR       = f"/mnt/home/jlee2/camels/{SUBGRID}/"
        RESULT_DIR     = f"/mnt/home/jlee2/camels/{SUBGRID}/results/{SUBGRID}/"
        LABEL_FILENAME = BASE_DIR + f"CosmoAstroSeed_{SUBGRID}_L25n256_LH.txt"
    elif TYPE == "CAMELS_50":
        BASE_DIR       = f"/mnt/home/jlee2/ceph/camels/SB35/{SUBGRID}/"
        DATA_DIR       = f"/mnt/home/jlee2/ceph/camels/SB35/{SUBGRID}/"
        RESULT_DIR     = f"/mnt/home/jlee2/ceph/camels/SB35/{SUBGRID}/results/{SUBGRID}/"
        LABEL_FILENAME = BASE_DIR + f"CosmoAstroSeed_{SUBGRID}_L50n512_SB35.txt"
    else:
        raise Exception("Invalid Simulation Suite")
else:
    raise Exception("Invalid Machine")

if TYPE == "Quijote":
    CATALOG_SIZE = 2000 
elif TYPE == "Quijote_Rockstar":
    CATALOG_SIZE = 3072
elif TYPE == "fR":
    CATALOG_SIZE = 2048
elif TYPE == "CAMELS":
    CATALOG_SIZE = 1000
elif TYPE == "CAMELS_50":
    CATALOG_SIZE = 1024
