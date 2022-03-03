import sys
import json
import numpy as np

# todo :
# optional post
# dump csv on keyboard interupt


if len(sys.argv) != 3:
    print("Requires a congig file (json and an ouput file (csv)")
    sys.exit(1)

ofn = sys.argv[2]

config = {}
with open(sys.argv[1]) as f:
    config = json.load(f)

ppnames = config["prep"]["parameters"].keys()
PPs = []
TPPs = [{}]
for i, p in enumerate(ppnames):
    PPs = TPPs.copy()
    TPPs = []
    cp = config["prep"]["parameters"][p]
    for v in np.arange(cp["min"], cp["max"]+cp["inc"], cp["inc"]):
        for ep in PPs:
            nep = ep.copy()
            nep[p] = v
            TPPs.append(nep)
PPs = TPPs

mpnames = config["model"]["parameters"].keys()
MPs = []
TMPs = [{}]
for i, p in enumerate(mpnames):
    MPs = TMPs.copy()
    TMPs = []
    cp = config["model"]["parameters"][p]
    for v in np.arange(cp["min"], cp["max"]+cp["inc"], cp["inc"]):
        for ep in MPs:
            nep = ep.copy()
            nep[p] = v
            TMPs.append(nep)
MPs = TMPs

opnames = config["post"]["parameters"].keys()
OPs = []
OMPs = [{}]
for i, p in enumerate(opnames):
    OPs = OMPs.copy()
    OMPs = []
    cp = config["post"]["parameters"][p]
    for v in np.arange(cp["min"], cp["max"]+cp["inc"], cp["inc"]):
        for ep in OPs:
            nep = ep.copy()
            nep[p] = v
            OMPs.append(nep)
OPs = OMPs

prepcode = ""
with open(config["prep"]["code"]) as f:
    prepcode = f.read()

modelcode = ""
with open(config["model"]["code"]) as f:
    modelcode = f.read()

postcode = ""
with open(config["post"]["code"]) as f:
    postcode = f.read()
    
results = []
for PP in PPs:
    P = PP.copy()
    print(P)
    exec(prepcode)
    for MP in MPs:
        P = MP.copy()
        print(P)
        exec(modelcode)
        for OP in OPs:
            P = OP.copy()
            print(P)
            exec(postcode)
            ev = {}
            for e in config["eval"]:
                ec = config["eval"][e]
                exec (f"val = {ec}")
                ev[e] = val
            print(ev)
            results.append({"PP": PP, "MP": MP, "OP": OP, "E": ev})

enames = config["eval"].keys()

fl = ""
for p in ppnames:
    if fl != "": fl+=","
    fl+=p
for p in mpnames:
    if fl != "": fl+=","
    fl+=p
for p in opnames:
    if fl != "": fl+=","
    fl+=p    
for p in enames:
    if fl != "": fl+=","
    fl+=p

with open(ofn, "w") as f:
    f.write(fl+"\n")
    for r in results:
        l = ""
        for p in ppnames:
            if l != "": l+=","
            l+=str(r["PP"][p])
        for p in mpnames:
            if l != "": l+=","
            l+=str(r["MP"][p])
        for p in opnames:
            if l != "": l+=","
            l+=str(r["OP"][p])            
        for p in enames:
            if l != "": l+=","
            l+=str(r["E"][p])
        f.write(l+"\n")
